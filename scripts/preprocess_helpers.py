"""
Shared helpers for preprocess_hest_stomics.

Note: Adapted from existing notebook/script utilities; concentrated here to keep the main script concise.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from aiohttp import ClientSession
from scipy import sparse
from scipy.spatial import cKDTree
from sentence_transformers import SentenceTransformer

ENSEMBL_RE = re.compile(r"^ENSG[0-9]+", re.IGNORECASE)
GRCH_PREFIX_RE = re.compile(r"^grch38_+")


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


async def translate_ensembl_ids(ensembl_ids, logger: logging.Logger, concurrency: int = 10) -> Dict[str, str]:
    """Translate Ensembl gene IDs to symbols using mygene asynchronously."""
    translation = {}
    semaphore = asyncio.Semaphore(concurrency)
    async with ClientSession() as session:
        tasks = [
            _fetch_symbol(session, gene_id, semaphore, logger)
            for gene_id in ensembl_ids
        ]
        results = await asyncio.gather(*tasks)
    for gene_id, symbol in zip(ensembl_ids, results):
        if symbol:
            translation[gene_id.upper()] = symbol
    return translation


def normalize_gene_names(var_names, translation_map, logger: logging.Logger) -> Tuple[List[str], List[int]]:
    """Apply gene normalization and drop untranslatable Ensembl IDs."""
    processed: List[str] = []
    keep_idx: List[int] = []
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
        return np.array([]), np.array([])
    return X.min(axis=1), X.max(axis=1), X.mean(axis=1)


def build_embeddings(texts, logger: logging.Logger):
    """Encode texts with SentenceTransformer and ensure uniqueness."""
    if not texts:
        return {}, None, []
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
    return embedding_dict, model_id, unique_texts


def _get_pixel_size_um(adata: ad.AnnData):
    for key in ["pixel_size", "pixel_size_um_estimated", "pixel_size_um_embedded"]:
        if key in adata.uns:
            return adata.uns[key]
    return None


def _get_pixel_size_from_meta(meta_row: pd.Series):
    for key in ["pixel_size_um_embedded", "pixel_size_um_estimated"]:
        if key in meta_row and pd.notna(meta_row[key]):
            return meta_row[key]
    return None


def nn_spacing_check(adata: ad.AnnData, pixel_size_um: float, st_tech: str, logger: logging.Logger, split_title: str):
    coords = np.asarray(adata.obsm["spatial"])
    if coords.shape[0] < 2:
        return
    tree = cKDTree(coords)
    dists_px, _ = tree.query(coords, k=2)
    nn_px = dists_px[:, 1]
    if pixel_size_um:
        nn_um = nn_px * float(pixel_size_um)
        median_um = float(np.median(nn_um))
        median_px = float(np.median(nn_px))
        expected = None
        st_lower = st_tech.lower() if isinstance(st_tech, str) else ""
        if "xenium" in st_lower:
            expected = 100.0
        elif "visium hd" in st_lower:
            expected = 128.0
        elif "visium" in st_lower:
            expected = 100.0  # center-to-center
        if expected:
            if abs(median_um - expected) / expected > 0.15:
                logger.warning(
                    "%s: NN spacing median %.2f um deviates from expected %.1f um (median px=%.2f)",
                    split_title,
                    median_um,
                    expected,
                    median_px,
                )
            else:
                logger.info(
                    "%s: NN spacing median=%.2f um (expected ~%.1f, px median=%.2f)",
                    split_title,
                    median_um,
                    expected,
                    median_px,
                )
        else:
            logger.info(
                "%s: NN spacing median=%.2f um (pixels median=%.2f)",
                split_title,
                median_um,
                median_px,
            )
    else:
        logger.info("%s: NN spacing median=%.2f px (pixel size unknown)", split_title, float(np.median(nn_px)))
