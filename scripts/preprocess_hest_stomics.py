import argparse
import asyncio
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree
from tqdm import tqdm

# ensure local imports work when running as a script
import sys
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
from pseudo_visium_fixed import (  # type: ignore
    pool_bins_visiumhd_fixed,
    dump_patches_fixed,
)
from preprocess_helpers import (
    setup_logging,
    translate_ensembl_ids,
    normalize_gene_names,
    _axis_stats,
    build_embeddings,
    _get_pixel_size_um,
    _get_pixel_size_from_meta,
    ENSEMBL_RE,
    VISIUM_HD_POOL_FACTOR,
    VISIUM_HD_DST_BIN_UM,
    XENIUM_SPOT_UM,
)



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
        default=Path("/storage/hest1k/HEST_v1_2_1.csv"),
        help="Path to metadata CSV.",
    )
    parser.add_argument(
        "--st-path",
        type=Path,
        default=Path("/storage/hest1k/st"),
        help="Path to directory containing ST .h5ad files.",
    )
    parser.add_argument(
        "--tif-root",
        type=Path,
        default=Path("/storage/hest1k/wsis"),
        help="Root directory where slide .tif files are stored (expected as <id>.tif). If missing, patches are skipped.",
    )
    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    logger.info("Starting preprocessing with metadata: %s | st_path: %s", args.metadata, args.st_path)
    meta_df = pd.read_csv(args.metadata)

    processed_dir = args.st_path.parent / f"processed_{datetime.now():%Y%m%d}"
    processed_dir.mkdir(parents=True, exist_ok=True)
    h5ad_dir = processed_dir / "h5ad"
    h5ad_dir.mkdir(exist_ok=True)
    patch_dir = processed_dir / "patches"
    patch_dir.mkdir(exist_ok=True)
    emb_dir = processed_dir / "embeddings"
    emb_dir.mkdir(exist_ok=True)
    logger.info("Processed .h5ad outputs will be saved to: %s", processed_dir)

    # slice to Homo sapiens and exclude Spatial Transcriptomics technology
    human_df = meta_df[
        (meta_df["species"] == "Homo sapiens") & (meta_df["st_technology"] != "Spatial Transcriptomics")
    ].copy()
    if human_df.empty:
        raise ValueError("No datasets left after filtering by species/technology.")
    if "organ" in human_df.columns:
        organ_counts = human_df["organ"].value_counts(dropna=False)
        logger.info("Datasets per organ after filtering:\n%s", organ_counts.to_string())

    # precompute metadata index for quick lookup
    meta_lookup = human_df.set_index("id")

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
            if sample_id not in meta_lookup.index:
                logger.warning("%s: missing metadata row; skipping", sample_id)  # note to self: check in the logs if this ever happens
                skip_records["missing_meta"].append(sample_id)
                continue
            meta_row = meta_lookup.loc[sample_id]
            estimated_px = None
            if "pixel_size_um_estimated" in adata.uns and adata.uns["pixel_size_um_estimated"] is not None:
                estimated_px = adata.uns["pixel_size_um_estimated"]
            elif "pixel_size_um_estimated" in meta_row and pd.notna(meta_row["pixel_size_um_estimated"]):
                estimated_px = meta_row["pixel_size_um_estimated"]
            if estimated_px is None:
                raise ValueError(f"{sample_id}: missing pixel_size_um_estimated in adata.uns and metadata")
            if "pixel_size_um_estimated" not in adata.uns or adata.uns["pixel_size_um_estimated"] is None:
                adata.uns["pixel_size_um_estimated"] = estimated_px
            st_tech = str(meta_row.get("st_technology", ""))
            # Pseudo-Visium pooling for Visium HD when possible
            if "visium hd" in st_tech.lower():
                adata.obs["spot_size_um"] = VISIUM_HD_DST_BIN_UM
                pixel_size = _get_pixel_size_um(adata) or _get_pixel_size_from_meta(meta_row)
                if pixel_size is None:
                    logger.warning("%s: Visium HD pooling skipped (missing pixel size)", sample_id)
                    skip_records["hd_no_pixel"].append(sample_id)
                elif not {"pxl_row_in_fullres", "pxl_col_in_fullres"}.issubset(adata.obs.columns):
                    logger.warning("%s: Visium HD pooling skipped (missing spatial columns)", sample_id)
                    skip_records["hd_no_spatial"].append(sample_id)
                else:
                    logger.info(
                        "%s: pooling Visium HD to pseudo-Visium (target %dum = %dx16um)",
                        sample_id,
                        VISIUM_HD_DST_BIN_UM,
                        VISIUM_HD_POOL_FACTOR,
                    )
                    expected_px = VISIUM_HD_DST_BIN_UM / pixel_size
                    logger.info("%s: target spot diameter ~%.2f px at pixel size %.4f um/px", sample_id, expected_px, pixel_size)
                    adata = pool_bins_visiumhd_fixed(adata, pixel_size=pixel_size, dst_bin_size_um=VISIUM_HD_DST_BIN_UM)
                    adata.obs["spot_size_um"] = VISIUM_HD_DST_BIN_UM
            elif "xenium" in st_tech.lower():
                # HEST Xenium h5ad is already spot-level with spatial coords; no pooling needed
                logger.info("%s: Xenium detected; using provided spots (no pooling)", sample_id)
                adata.obs["spot_size_um"] = XENIUM_SPOT_UM
            dup_mask = adata.var_names.duplicated()
            dup_count = int(dup_mask.sum())
            if dup_count:
                dup_names = adata.var_names[dup_mask][:5].tolist()  # up to 5 to avoid cluttering the logs
                logger.info("%s: dropping %d duplicate var_names -> %s", sample_id, dup_count, dup_names)
                adata = adata[:, ~dup_mask].copy()
            ensembl_candidates.update([g for g in adata.var_names if ENSEMBL_RE.match(g)])
            adata.obs["dataset_title"] = dataset_title
            adata.obs["sample_id"] = sample_id
            adata.obs["st_technology"] = meta_row.get("st_technology")
            adata.obs["organ"] = meta_row.get("organ")
            adata.obs["preservation_method"] = meta_row.get("preservation_method")
            adata.obs["pixel_size_um_meta"] = _get_pixel_size_from_meta(meta_row)
            adata.obs["spot_diameter_meta"] = meta_row.get("spot_diameter")
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
    spots_records = []
    skip_records = {"missing_meta": [], "hd_no_pixel": [], "hd_no_spatial": [], "no_tif": [], "patch_fail": []}
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
                logger.info("%s: empty gene panel -> dropping split", split_title)  # check if this ever happens
                dropped_datasets.append((split_title, 100.0))
                continue

            adatas_common = [a[:, common_genes].copy() for a in split_adatas]
            concat = ad.concat(adatas_common, join="inner", keys=split_ids, label="sample_id")

            gene_stats = _axis_stats(concat.X, axis=0, stats=("min", "max", "mean"))
            constant_mask = (gene_stats["max"] - gene_stats["min"]) == 0
            genes_to_drop = concat.var_names[constant_mask].tolist()

            filtered_genes[split_title] = genes_to_drop
            drop_pct = 100 * len(genes_to_drop) / len(concat.var_names) if len(concat.var_names) else 0
            logger.info(
                "%s: drop %d genes (%d constant, %.1f%%) -> %s ... (max 5 shown)",
                split_title,
                len(genes_to_drop),
                int(constant_mask.sum()),
                drop_pct,
                genes_to_drop[:5],
            )

            if drop_pct > 90:
                # check if this ever happens
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
                meta_row = meta_lookup.loc[sid]
                spot_ids = [f"{sid}__{obs}" for obs in filtered_adata.obs_names]
                filtered_adata.obs["spot_id"] = spot_ids
                filtered_adata.obs_names = spot_ids
                out_path = h5ad_dir / f"{sid}.h5ad"
                filtered_adata.write(out_path)
                logger.info("%s: saved processed h5ad -> %s (genes kept: %d)", sid, out_path, filtered_adata.n_vars)

                # Patch extraction if tif exists
                tif_path = None
                for ext in [".tif", ".tiff"]:
                    candidate = args.tif_root / f"{sid}{ext}"
                    if candidate.exists():
                        tif_path = candidate
                        break
                patch_path = None
                if tif_path is not None and filtered_adata.shape[0] > 0:
                    # prefer uns pixel size; fallback to metadata column (positional access to avoid FutureWarning)
                    pixel_size_meta = None
                    if "pixel_size_um_meta" in filtered_adata.obs:
                        try:
                            pixel_size_meta = filtered_adata.obs["pixel_size_um_meta"].iloc[0]
                        except Exception:
                            pixel_size_meta = None
                    pixel_size = _get_pixel_size_um(filtered_adata) or pixel_size_meta
                    if pixel_size is not None:
                        filtered_adata.uns["pixel_size"] = pixel_size
                        patch_fov_um = 224 * 0.5  # target_patch_size * target_pixel_size
                        resample_ratio = 0.5 / float(pixel_size)
                        logger.info(
                            "%s: patch FOV %.1f um @ target_pixel_size=0.5; resample ratio from src %.3f",
                            sid,
                            patch_fov_um,
                            resample_ratio,
                        )
                    patch_path = patch_dir / f"slide={sid}_images.npz"
                    oob_log_path = processed_dir / "patch_oob.log"
                    dump_patches_fixed(
                        filtered_adata,
                        tif_path,
                        patch_path.parent,
                        name=f"slide={sid}",
                        oob_log_path=oob_log_path,
                    )
                    logger.info("%s: saved patches -> %s", sid, patch_path)
                else:
                    if tif_path is None:
                        skip_records["no_tif"].append(sid)

                # Record spots metadata
                pixel_size_um = _get_pixel_size_um(filtered_adata)
                for idx, spot in filtered_adata.obs.iterrows():
                    spots_records.append(
                        {
                            "spot_id": idx,
                            "slide_id": sid,
                            "dataset_title": dataset_title,
                            "organ": spot.get("organ"),
                            "st_technology": spot.get("st_technology"),
                            "preservation_method": spot.get("preservation_method"),
                            "pixel_x": float(filtered_adata.obsm["spatial"][filtered_adata.obs_names.get_loc(idx)][0]),
                            "pixel_y": float(filtered_adata.obsm["spatial"][filtered_adata.obs_names.get_loc(idx)][1]),
                            "patch_file": str(patch_path) if patch_path else None,
                            "h5ad_file": str(out_path),
                            "pixel_size_um": pixel_size_um,
                            "tif_path": str(tif_path) if tif_path else None,
                        }
                    )
                filtered_adatas.append(filtered_adata)

            concat_filtered = ad.concat(filtered_adatas, join="inner", keys=split_ids, label="sample_id")

            # per-gene stats (min/q1/median/q3/max/mean across spots)
            gene_stats = _axis_stats(
                concat_filtered.X,
                axis=0,
                stats=("min", "max", "mean", "q1", "median", "q3"),
            )
            gene_stats_df = pd.DataFrame(
                {
                    "gene_min": gene_stats["min"],
                    "gene_q1": gene_stats["q1"],
                    "gene_median": gene_stats["median"],
                    "gene_q3": gene_stats["q3"],
                    "gene_max": gene_stats["max"],
                    "gene_mean": gene_stats["mean"],
                },
                index=concat_filtered.var_names,
            )
            logger.info(
                "%s: per-gene stats (min/q1/median/q3/max/mean across spots) describe ->\n%s",
                split_title,
                gene_stats_df.describe(),
            )

            # per-spot stats (min/max/mean across genes)
            spot_stats = _axis_stats(concat_filtered.X, axis=1, stats=("min", "max", "mean"))
            if spot_stats["min"].size == 0:
                logger.info("%s: no genes left after filtering; skipping stats", split_title)
                dropped_datasets.append((split_title, 100.0))
                continue

            spot_stats_df = pd.DataFrame(
                {
                    "spot_min": spot_stats["min"],
                    "spot_max": spot_stats["max"],
                    "spot_mean": spot_stats["mean"],
                },
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

            # sanity checks: nearest-neighbor spacing (per slide)
            for sid, slide_adata in zip(split_ids, filtered_adatas):
                pixel_size_um = _get_pixel_size_um(slide_adata)
                if pixel_size_um is None:
                    raise ValueError(f"{sid}: missing pixel_size_um_estimated for NN spacing check")
                try:
                    coords = np.asarray(slide_adata.obsm["spatial"])
                    if coords.shape[0] < 2:
                        continue
                    tree = cKDTree(coords)
                    dists_px, _ = tree.query(coords, k=2)
                    nn_px = dists_px[:, 1]
                    nn_um = nn_px * float(pixel_size_um)
                    median_um = float(np.median(nn_um))
                    median_px = float(np.median(nn_px))
                    st_tech = str(slide_adata.obs["st_technology"].iloc[0]).lower()
                    expected = None
                    if "xenium" in st_tech:
                        expected = XENIUM_SPOT_UM
                    elif "visium hd" in st_tech:
                        expected = VISIUM_HD_DST_BIN_UM
                    elif "visium" in st_tech:
                        expected = 100.0  # center-to-center
                    if expected:
                        if abs(median_um - expected) / expected > 0.15:
                            logger.warning(
                                "%s (%s): NN spacing median %.2f um deviates from expected %.1f um (median px=%.2f)",
                                split_title,
                                sid,
                                median_um,
                                expected,
                                median_px,
                            )
                        else:
                            logger.info(
                                "%s (%s): NN spacing median=%.2f um (expected ~%.1f, px median=%.2f)",
                                split_title,
                                sid,
                                median_um,
                                expected,
                                median_px,
                            )
                    else:
                        logger.info(
                            "%s (%s): NN spacing median=%.2f um (pixels median=%.2f)",
                            split_title,
                            sid,
                            median_um,
                            median_px,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s (%s): NN spacing check failed (%s)", split_title, sid, exc)

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
    embeddings_dict, model_used, embedding_texts = build_embeddings(embedding_inputs, logger)
    if embeddings_dict:
        logger.info("Built embeddings for %d texts using model %s", len(embeddings_dict), model_used)
        emb_matrix = np.stack([embeddings_dict[t] for t in embedding_texts])
        np.save(emb_dir / "embeddings.npy", emb_matrix)
        with open(emb_dir / "key_to_index.json", "w", encoding="utf-8") as f:
            json.dump({t: i for i, t in enumerate(embedding_texts)}, f)
        with open(emb_dir / "index_to_text.json", "w", encoding="utf-8") as f:
            json.dump({str(i): t for i, t in enumerate(embedding_texts)}, f)
        with open(emb_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_id": model_used,
                    "normalize": True,
                    "num_texts": len(embedding_texts),
                    "dtype": str(emb_matrix.dtype),
                },
                f,
                indent=2,
            )
    else:
        logger.info("No embeddings built (no texts).")

    if spots_records:
        spots_df = pd.DataFrame(spots_records)
        spots_df.to_parquet(processed_dir / "spots.parquet", index=False)
        logger.info("Saved spots metadata -> %s", processed_dir / "spots.parquet")

    # log skipped items summary
    sep = "-" * 80
    logger.info(sep)
    logger.info("Skip summary:")
    for key, vals in skip_records.items():
        logger.info(" - %s: %d", key, len(vals))
    logger.info(sep)

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
