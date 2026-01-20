import argparse
import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from aiohttp import ClientSession
from scipy import sparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def setup_logging(log_file: Path) -> logging.Logger:
    """Configure root logger to log to stdout and file."""
    logger = logging.getLogger("gene_filter")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


ENSEMBL_RE = re.compile(r"^ENSG[0-9]+", re.IGNORECASE)
GRCH_PREFIX_RE = re.compile(r"^grch38_+")


async def _fetch_symbol(session: ClientSession, gene_id: str, semaphore: asyncio.Semaphore, logger: logging.Logger):
    url = "https://mygene.info/v3/query"
    params = {"q": gene_id, "scopes": "ensembl.gene", "fields": "symbol", "species": "human"}
    async with semaphore:
        try:
            async with session.get(url, params=params, timeout=15) as resp:
                if resp.status != 200:
                    logger.debug("mygene %s status %s", gene_id, resp.status)
                    return None
                data = await resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.debug("mygene %s request failed: %s", gene_id, exc)
            return None
    hits = data.get("hits") or []
    for hit in hits:
        symbol = hit.get("symbol")
        if symbol:
            return symbol
    return None


async def translate_ensembl_ids(ensembl_ids, logger: logging.Logger, concurrency: int = 10):
    """Translate Ensembl gene IDs to symbols using mygene asynchronously."""
    translation = {}
    semaphore = asyncio.Semaphore(concurrency)
    connector = None
    async with ClientSession(connector=connector) as session:
        tasks = [
            _fetch_symbol(session, gene_id, semaphore, logger)
            for gene_id in ensembl_ids
        ]
        results = await asyncio.gather(*tasks)
    for gene_id, symbol in zip(ensembl_ids, results):
        if symbol:
            translation[gene_id.upper()] = symbol
    return translation


def normalize_gene_names(var_names, translation_map, logger: logging.Logger):
    """Apply gene normalization and drop untranslatable Ensembl IDs."""
    processed = []
    keep_idx = []
    seen = set()
    dropped_untranslated = 0
    for idx, name in enumerate(var_names):
        gene = name
        if ENSEMBL_RE.match(gene):
            symbol = translation_map.get(gene.upper())
            if not symbol:
                dropped_untranslated += 1
                continue
            gene = symbol
        gene = gene.lower()
        gene = GRCH_PREFIX_RE.sub("", gene)
        if gene and gene not in seen:
            seen.add(gene)
            processed.append(gene)
            keep_idx.append(idx)
    if dropped_untranslated:
        logger.info("Dropped %d Ensembl genes lacking translation", dropped_untranslated)
    return processed, keep_idx


def _gene_stats(X):
    if sparse.issparse(X):
        return (
            np.asarray(X.sum(axis=0)).ravel(),
            np.asarray(X.max(axis=0).todense()).ravel(),
            np.asarray(X.min(axis=0).todense()).ravel(),
        )
    X = np.asarray(X)
    return X.sum(axis=0), X.max(axis=0), X.min(axis=0)


def _spot_stats(X):
    if sparse.issparse(X):
        if X.shape[0] == 0 or X.shape[1] == 0:
            return np.array([]), np.array([]), np.array([])
        return (
            np.asarray(X.min(axis=1)).ravel(),
            np.asarray(X.max(axis=1)).ravel(),
            np.asarray(X.mean(axis=1)).ravel(),
        )
    X = np.asarray(X)
    if X.size == 0 or X.shape[1] == 0:
        return np.array([]), np.array([]), np.array([])
    return X.min(axis=1), X.max(axis=1), X.mean(axis=1)


def build_embeddings(texts, logger: logging.Logger):
    """Encode texts with SentenceTransformer and ensure uniqueness."""
    if not texts:
        return {}, None
    model_id = "thomas-sounack/BioClinical-ModernBERT-base"
    logger.info("Loading embedding model: %s", model_id)
    model = SentenceTransformer(model_id)
    unique_texts = list(dict.fromkeys(texts))  # preserve order and dedupe
    logger.info("Encoding %d unique texts", len(unique_texts))
    emb = model.encode(unique_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    emb_arr = np.asarray(emb)
    unique_rows = np.unique(emb_arr, axis=0).shape[0]
    if unique_rows != len(unique_texts):
        logger.warning("Embedding duplicates detected: %d unique embeddings vs %d texts", unique_rows, len(unique_texts))
    else:
        logger.info("All %d embeddings are unique", len(unique_texts))
    embedding_dict = {txt: vec for txt, vec in zip(unique_texts, emb_arr)}
    return embedding_dict, model_id


def main():
    parser = argparse.ArgumentParser(description="Preprocess HEST stomics datasets with panel splits and gene filtering.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/preprocess_hest_stomics.log"),
        help="Path to write logs.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/storage/hest1k/HEST_v1_1_0.csv"),
        help="Path to metadata CSV.",
    )
    parser.add_argument(
        "--st-path",
        type=Path,
        default=Path("/storage/hest1k/st"),
        help="Path to directory containing ST .h5ad files.",
    )
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    logger.info("Starting preprocessing with metadata: %s | st_path: %s", args.metadata, args.st_path)
    meta_df = pd.read_csv(args.metadata)

    processed_dir = args.st_path.parent / f"st_processed_{datetime.now():%Y%m%d}"
    processed_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Processed .h5ad outputs will be saved to: %s", processed_dir)

    # slice to Homo sapiens and exclude Spatial Transcriptomics technology
    human_df = meta_df[
        (meta_df["species"] == "Homo sapiens") & (meta_df["st_technology"] != "Spatial Transcriptomics")
    ].copy()
    if human_df.empty:
        raise ValueError("No datasets left after filtering by species/technology.")

    # load all h5ad files, dropping duplicate genes
    dataset_payloads = {}
    ensembl_candidates = set()
    for dataset_title, ids in tqdm(
        human_df.groupby("dataset_title")["id"].apply(list).items(),
        desc="Load adatas",
    ):
        adatas = []
        for sample_id in ids:
            adata = sc.read_h5ad(args.st_path / f"{sample_id}.h5ad")
            dup_mask = adata.var_names.duplicated()
            dup_count = int(dup_mask.sum())
            if dup_count:
                dup_names = adata.var_names[dup_mask][:5].tolist()  # up to 5 to avoid cluttering the logs
                logger.info("%s: dropping %d duplicate var_names -> %s", sample_id, dup_count, dup_names)
                adata = adata[:, ~dup_mask].copy()
            ensembl_candidates.update([g for g in adata.var_names if ENSEMBL_RE.match(g)])
            adata.obs["dataset_title"] = dataset_title
            adata.obs["sample_id"] = sample_id
            adatas.append(adata)
        dataset_payloads[dataset_title] = {"ids": ids, "adatas": adatas}

    translation_map = {}
    if ensembl_candidates:
        logger.info("Translating %d Ensembl gene IDs via mygene", len(ensembl_candidates))
        try:
            translation_map = asyncio.run(translate_ensembl_ids(list(ensembl_candidates), logger))
            logger.info("Translated %d/%d Ensembl IDs", len(translation_map), len(ensembl_candidates))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ensembl translation failed (%s); continuing without translations", exc)
            translation_map = {}
        if len(translation_map) == 0:
            logger.warning("Ensembl translation returned zero results; proceeding but untranslated Ensembl genes will be dropped.")

    rng = np.random.default_rng(0)
    filtered_genes = defaultdict(list)
    value_summaries = {}
    dropped_datasets = []
    split_report = {}
    surviving_genes = set()

    # normalize gene names and drop untranslated Ensembl genes
    for dataset_title, payload in dataset_payloads.items():
        processed_adatas = []
        for adata in payload["adatas"]:
            processed_names, keep_idx = normalize_gene_names(adata.var_names, translation_map, logger)
            adata = adata[:, keep_idx].copy()
            adata.var_names = processed_names
            processed_adatas.append(adata)
        payload["adatas"] = processed_adatas

    # per-dataset, split by identical gene panels
    for dataset_title, payload in tqdm(dataset_payloads.items(), desc="Split/filter genes"):
        ids = payload["ids"]
        adatas = payload["adatas"]

        panel_groups = {}
        for i, adata in enumerate(adatas):
            key = tuple(sorted(adata.var_names))  # order-insensitive panel signature
            panel_groups.setdefault(key, []).append(i)

        num_splits = len(panel_groups)
        pct_splits = 100.0 * num_splits / len(adatas)
        split_report[dataset_title] = {"num_splits": num_splits, "pct_splits": pct_splits}
        logger.info("%s: %d split(s); %.1f%% relative to %d tissues", dataset_title, num_splits, pct_splits, len(adatas))

        for split_idx, (panel_key, idxs) in enumerate(panel_groups.items(), start=1):
            split_title = dataset_title if (num_splits == 1) else f"(Split {split_idx}) {dataset_title}"
            split_ids = [ids[i] for i in idxs]
            split_adatas = [adatas[i] for i in idxs]

            common_genes = list(panel_key)  # identical by construction
            if not common_genes:
                logger.info("%s: empty gene panel -> dropping split", split_title)
                dropped_datasets.append((split_title, 100.0))
                continue

            adatas_common = [a[:, common_genes].copy() for a in split_adatas]
            concat = ad.concat(adatas_common, join="inner", keys=split_ids, label="sample_id")

            sums, maxs, mins = _gene_stats(concat.X)
            constant_mask = (maxs - mins) == 0
            genes_to_drop = concat.var_names[constant_mask].tolist()

            filtered_genes[split_title] = genes_to_drop
            drop_pct = 100 * len(genes_to_drop) / len(concat.var_names) if len(concat.var_names) else 0
            logger.info(
                "%s: drop %d genes (%d constant, %.1f%%) -> %s (max 5 shown)",
                split_title,
                len(genes_to_drop),
                int(constant_mask.sum()),
                drop_pct,
                genes_to_drop[:5],
            )

            if drop_pct > 90:
                logger.info("%s: dropping split because %.1f%% of genes would be removed", split_title, drop_pct)
                dropped_datasets.append((split_title, drop_pct))
                continue

            keep_mask = ~constant_mask
            kept_genes = concat.var_names[keep_mask]
            if len(kept_genes) == 0:
                logger.info("%s: no genes left after filtering; skipping stats/save", split_title)
                dropped_datasets.append((split_title, 100.0))
                continue
            surviving_genes.update(kept_genes)

            # Save per-slide filtered adatas with original dataset title
            filtered_adatas = []
            for adata_proc, sid in zip(adatas_common, split_ids):
                filtered_adata = adata_proc[:, keep_mask].copy()
                filtered_adata.obs["dataset_title"] = dataset_title
                out_path = processed_dir / f"{sid}.h5ad"
                filtered_adata.write(out_path)
                logger.info("%s: saved processed h5ad -> %s (genes kept: %d)", sid, out_path, filtered_adata.n_vars)
                filtered_adatas.append(filtered_adata)

            concat_filtered = ad.concat(filtered_adatas, join="inner", keys=split_ids, label="sample_id")

            # per-gene stats (min/q1/median/q3/max/mean across spots)
            vals = concat_filtered.X.toarray() if sparse.issparse(concat_filtered.X) else np.asarray(concat_filtered.X)
            gene_min = vals.min(axis=0)
            gene_max = vals.max(axis=0)
            gene_mean = vals.mean(axis=0)
            gene_q1 = np.percentile(vals, 25, axis=0)
            gene_median = np.percentile(vals, 50, axis=0)
            gene_q3 = np.percentile(vals, 75, axis=0)
            gene_stats_df = pd.DataFrame(
                {
                    "gene_min": gene_min,
                    "gene_q1": gene_q1,
                    "gene_median": gene_median,
                    "gene_q3": gene_q3,
                    "gene_max": gene_max,
                    "gene_mean": gene_mean,
                },
                index=concat_filtered.var_names,
            )
            logger.info(
                "%s: per-gene stats (min/q1/median/q3/max/mean across spots) describe ->\n%s",
                split_title,
                gene_stats_df.describe(),
            )

            # per-spot stats (min/max/mean across genes)
            spot_min, spot_max, spot_mean = _spot_stats(concat_filtered.X)
            if spot_min.size == 0:
                logger.info("%s: no genes left after filtering; skipping stats", split_title)
                dropped_datasets.append((split_title, 100.0))
                continue

            spot_stats_df = pd.DataFrame(
                {"spot_min": spot_min, "spot_max": spot_max, "spot_mean": spot_mean},
                index=concat_filtered.obs_names,
            )
            logger.info(
                "%s: per-spot stats (min/max/mean across genes) describe ->\n%s",
                split_title,
                spot_stats_df.describe(),
            )

            rand_idx = int(rng.integers(0, len(split_ids)))
            rand_id = split_ids[rand_idx]
            rand_adata = filtered_adatas[rand_idx]
            vals = rand_adata.X.toarray() if sparse.issparse(rand_adata.X) else np.asarray(rand_adata.X)
            flat_vals = vals.ravel()
            summary = {
                "min": float(np.min(flat_vals)),
                "q1": float(np.percentile(flat_vals, 25)),
                "median": float(np.median(flat_vals)),
                "q3": float(np.percentile(flat_vals, 75)),
                "max": float(np.max(flat_vals)),
                "mean": float(np.mean(flat_vals)),
            }
            value_summaries[split_title] = {"sample_id": rand_id, "summary": summary}
            logger.info("%s: random tissue %s value summary %s", split_title, rand_id, summary)

    # Build embeddings for organs, st technology, tissue prep methods, and surviving genes (the ones that remain after filtering)
    def _collect_unique(df: pd.DataFrame, cols):
        values = []
        for col in cols:
            if col in df.columns:
                values.extend([str(x) for x in df[col].dropna().unique()])
        return list(dict.fromkeys(values))  # dedupe, preserve order

    organ_texts = _collect_unique(human_df, ["organ"])
    tech_texts = _collect_unique(human_df, ["st_technology"])
    prep_texts = _collect_unique(human_df, ["preservation_method"])
    gene_texts = sorted(surviving_genes)
    embedding_inputs = organ_texts + tech_texts + prep_texts + gene_texts
    embeddings_dict, model_used = build_embeddings(embedding_inputs, logger)
    if embeddings_dict:
        logger.info("Built embeddings for %d texts using model %s", len(embeddings_dict), model_used)
    else:
        logger.info("No embeddings built (no texts).")

    # final report of dropped datasets/splits and gene drops
    sep = "-" * 80
    logger.info(sep)
    logger.info("Dropped datasets/splits (>90%% genes removed or no genes left):")
    if not dropped_datasets:
        logger.info("None")
    else:
        for ds, pct in dropped_datasets:
            logger.info(" - %s: %.1f%% genes would be removed", ds, pct)
    logger.info(sep)

    logger.info("Gene drops by split (count only):")
    if not filtered_genes:
        logger.info("None")
    else:
        for split_title, genes in filtered_genes.items():
            logger.info(" - %s: %d genes dropped", split_title, len(genes))
    logger.info(sep)

    logger.info("Surviving genes across all processed splits: %d", len(surviving_genes))

    logger.info("Split summary by dataset:")
    for ds, info in split_report.items():
        logger.info(" - %s: %d split(s), %.1f%% relative to %d tissues", ds, info["num_splits"], info["pct_splits"], len(dataset_payloads[ds]["adatas"]))
    logger.info(sep)


if __name__ == "__main__":
    main()
