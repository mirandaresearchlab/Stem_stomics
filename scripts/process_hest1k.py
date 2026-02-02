#!/usr/bin/env python3
"""Process HEST1K samples into image patch embeddings and (optionally) select genes.

This script re-uses the logic from `dataset_preprocess.ipynb` to: 
- load ST AnnData and corresponding H&E `*.tif` images,
- extract per-spot image patch embeddings using CONCH and UNI (and optionally other models),
- save processed embeddings under `processed_data/` in the configured save path,
- optionally compute and save two gene lists (HVG and HMHVG) across processed slides.

Configuration is provided via a TOML file and validated with a Pydantic model. Use `-c / --config` to point to
the TOML file. See `scripts/README_process_hest1k.md` for a minimal example.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Optional, Literal
from datetime import datetime

import torch
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import anndata
import pandas as pd

from pydantic import BaseModel, Field
from utils.config_loader import load_toml_config


def setup_logging(save_path: Path, log_file: str = "process_hest1k.log") -> logging.Logger:
    """Set up logging to both console and file."""
    log_path = save_path / "processed_data" / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


class QueryConfig(BaseModel):
    # If csv_path is None, the script will try to load metadata from the HEST dataset on HuggingFace.
    csv_path: Optional[Path] = None
    column: str
    method: Literal["startswith", "contains", "equals"] = "startswith"
    pattern: str
    # optional exact-match filters: {"disease_state": "Healthy"}
    filters: Optional[dict] = None


class PreprocessConfig(BaseModel):
    # filesystem
    data_path: Path
    tif_path: Optional[Path] = None
    st_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    save_path: Optional[Path] = None

    # sample selection: either a direct list/file or a query against the metadata CSV
    selection_mode: Literal["list", "query"] = "list"
    # list mode
    ids_to_query: Optional[List[str]] = None
    ids_file: Optional[Path] = None
    # query mode
    ids_query: Optional[QueryConfig] = None

    # runtime
    device: str = "cuda"

    # embedding options
    models_to_run: List[str] = Field(default_factory=lambda: ["conch", "uni"])  # supported: conch, uni
    save_augmented: bool = True

    # gene selection (optional)
    run_gene_selection: bool = False
    hvg_top_k: int = 0           # size of final HVG list; 0 disables HVG output
    hmhvg_top_k: int = 0         # size of final HMHVG list; 0 disables HMHVG output
    deg_top_k: int = 0           # size per-group for DEG selection; 0 disables DEG output
    deg_groupby: Optional[str] = None  # obs column to group by for DEG selection
    deg_groups: Optional[List[str]] = None  # optional specific groups to test; defaults to all
    deg_label_dir: Optional[Path] = Path("data/ST-pat/lbl")  # optional: directory with per-slide region labels
    deg_subseries_column: str = "subseries"  # metadata column mapping id -> subseries (e.g., A1, B1)
    gene_list_filename: str = "selected_gene_list.txt"  # customize output filename for selected genes

    # error handling
    halt_on_error: bool = False  # If True, stop immediately on first error; if False, continue and collect failures
    
    # processing mode: 'both' (images + genes), 'images' (only image embeddings), 'genes' (only gene selection)
    process_mode: Literal["both", "images", "genes"] = "both"

def _resolve_ids(cfg: PreprocessConfig, logger: logging.Logger) -> List[str]:
    if cfg.selection_mode == "list":
        if cfg.ids_to_query:
            logger.info(f"Selection mode: list. Using ids_to_query with {len(cfg.ids_to_query)} samples.")
            return cfg.ids_to_query
        if cfg.ids_file:
            ids = [ln.strip() for ln in cfg.ids_file.read_text().splitlines() if ln.strip()]
            logger.info(f"Selection mode: list. Loaded {len(ids)} sample IDs from {cfg.ids_file}")
            return ids
        raise RuntimeError("Selection mode 'list' requires `ids_to_query` or `ids_file` in config")

    if cfg.selection_mode == "query":
        if cfg.ids_query is None:
            raise RuntimeError("Selection mode 'query' requires `ids_query` settings in config")
        q = cfg.ids_query
        # try to load metadata CSV from provided path or from HuggingFace dataset raw file
        if q.csv_path is not None:
            logger.info(f"Loading metadata CSV from: {q.csv_path}")
            meta_df = pd.read_csv(q.csv_path)
        else:
            hf_url = "https://huggingface.co/datasets/MahmoodLab/hest/resolve/main/HEST_v1_2_1.csv"
            logger.info(f"Loading metadata CSV from HuggingFace: {hf_url}")
            try:
                meta_df = pd.read_csv(hf_url)
            except Exception as e:
                logger.error(
                    "Could not load metadata CSV from HuggingFace. Provide `ids_query.csv_path` in config or ensure network access."
                )
                raise RuntimeError(
                    "Could not load metadata CSV from HuggingFace. Provide `ids_query.csv_path` in config or ensure network access."
                ) from e

        logger.info(f"Metadata CSV loaded with {len(meta_df)} rows")
        
        # optional exact-match filters
        if q.filters:
            logger.info(f"Applying filters: {q.filters}")
            for col, val in q.filters.items():
                meta_df = meta_df.loc[meta_df[col] == val]
            logger.info(f"After filters: {len(meta_df)} rows remain")

        if q.method == "startswith":
            ids = meta_df.loc[meta_df[q.column].str.startswith(q.pattern, na=False), "id"].tolist()
        elif q.method == "contains":
            ids = meta_df.loc[meta_df[q.column].str.contains(q.pattern, na=False), "id"].tolist()
        elif q.method == "equals":
            ids = meta_df.loc[meta_df[q.column] == q.pattern, "id"].tolist()
        else:
            raise RuntimeError(f"Unsupported query method: {q.method}")

        logger.info(f"Query matched {len(ids)} samples using {q.method}('{q.pattern}') on column '{q.column}'")
        return ids

    raise RuntimeError(f"Unknown selection_mode: {cfg.selection_mode}")


def ensure_dirs(save_path: Path):
    (save_path / "processed_data").mkdir(parents=True, exist_ok=True)
    (save_path / "processed_data/1spot_conch_ebd/").mkdir(parents=True, exist_ok=True)
    (save_path / "processed_data/1spot_uni_ebd/").mkdir(parents=True, exist_ok=True)
    (save_path / "processed_data/1spot_conch_ebd_aug/").mkdir(parents=True, exist_ok=True)
    (save_path / "processed_data/1spot_uni_ebd_aug/").mkdir(parents=True, exist_ok=True)


def _append_suffix_to_filename(filename: str, suffix: str) -> Path:
    """Return a filename with suffix inserted before the extension, preserving subdirectories."""
    base_path = Path(filename)
    new_name = f"{base_path.stem}{suffix}{base_path.suffix}"
    return base_path.with_name(new_name)


def _load_metadata_for_deg(cfg: PreprocessConfig, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Return metadata dataframe with id/subseries if available for DEG label mapping."""
    # Priority: explicit metadata_path; fallback to ids_query.csv_path if present
    meta_path = cfg.metadata_path or (cfg.ids_query.csv_path if cfg.selection_mode == "query" and cfg.ids_query and cfg.ids_query.csv_path else None)
    if meta_path is None:
        logger.warning("DEG: metadata_path not provided; cannot map subseries to ids for region labels")
        return None
    if not Path(meta_path).exists():
        logger.warning(f"DEG: metadata_path {meta_path} not found; cannot map subseries to ids for region labels")
        return None
    try:
        meta_df = pd.read_csv(meta_path)
    except Exception as e:
        logger.warning(f"DEG: failed to read metadata at {meta_path}: {e}")
        return None
    return meta_df


def _get_label_path_for_id(sid: str, cfg: PreprocessConfig, meta_df: Optional[pd.DataFrame]) -> Optional[Path]:
    """Resolve label file path for a given sample id using metadata subseries mapping."""
    if meta_df is None or cfg.deg_label_dir is None:
        return None
    if cfg.deg_subseries_column not in meta_df.columns or "id" not in meta_df.columns:
        return None
    subseries_rows = meta_df.loc[meta_df["id"] == sid, cfg.deg_subseries_column]
    if subseries_rows.empty:
        return None
    subseries = subseries_rows.iloc[0]
    return Path(cfg.deg_label_dir) / f"{subseries}_labeled_coordinates.tsv"


def _inject_region_labels_from_subseries(
    adata: anndata.AnnData,
    sid: str,
    cfg: PreprocessConfig,
    meta_df: Optional[pd.DataFrame],
    logger: logging.Logger,
    label_path: Optional[Path] = None,
) -> anndata.AnnData:
    """If deg_groupby column is missing, try to add it using subseries -> labeled_coordinates.tsv mapping."""
    groupby = cfg.deg_groupby or "region"
    if groupby in adata.obs:
        return adata

    if label_path is None:
        label_path = _get_label_path_for_id(sid, cfg, meta_df)

    if label_path is None:
        logger.info(f"DEG: no label path resolved for {sid}; skipping label injection")
        return adata
    if not label_path.exists():
        logger.info(f"DEG: label file not found for {sid}: {label_path}; skipping label injection")
        return adata

    try:
        lbl = pd.read_csv(label_path, sep="\t")
    except Exception as e:
        logger.warning(f"DEG: failed to read label file {label_path}: {e}")
        return adata
    if "Row.names" not in lbl.columns or "label" not in lbl.columns:
        logger.warning(f"DEG: label file {label_path} missing 'Row.names' or 'label' columns")
        return adata

    # Strategy 1: direct mapping using Row.names
    label_map = dict(zip(lbl["Row.names"], lbl["label"]))
    region_series_direct = adata.obs_names.map(label_map)
    direct_hits = int(region_series_direct.notna().sum())

    # Strategy 2: infer row/col keys from rounded x/y (label files use x≈row, y≈col)
    lbl_round = lbl.copy()
    # drop rows with non-finite coordinates before rounding
    lbl_round = lbl_round[pd.notna(lbl_round["x"]) & pd.notna(lbl_round["y"])]
    if not lbl_round.empty:
        lbl_round["row_r"] = lbl_round["x"].round().astype(int)
        lbl_round["col_r"] = lbl_round["y"].round().astype(int)
        lbl_round["key_rc"] = lbl_round["row_r"].map(lambda v: f"{v:03d}") + "x" + lbl_round["col_r"].map(lambda v: f"{v:03d}")
        label_map_round = dict(zip(lbl_round["key_rc"], lbl_round["label"]))
        region_series_round = adata.obs_names.map(label_map_round)
        round_hits = int(region_series_round.notna().sum())
    else:
        region_series_round = pd.Series([None] * adata.n_obs, index=adata.obs_names)
        round_hits = 0

    # choose better coverage
    if round_hits > direct_hits:
        region_series = region_series_round
        missing = int(region_series.isna().sum())
        logger.info(
            f"DEG: using rounded x/y mapping for {sid} ({round_hits} matches vs {direct_hits} direct)"
        )
    else:
        region_series = region_series_direct
        missing = int(region_series.isna().sum())
        if direct_hits > 0:
            logger.info(
                f"DEG: using Row.names mapping for {sid} ({direct_hits} matches)"
            )
        else:
            logger.info(f"DEG: no Row.names matches for {sid}")

    adata.obs[groupby] = region_series
    if missing > 0:
        logger.warning(f"DEG: {missing} spots in {sid} missing {groupby} labels after injection")
    else:
        logger.info(f"DEG: injected {groupby} labels for {sid} from {label_path}")
    return adata


def load_conch_and_uni(device: str = "cuda", logger: logging.Logger = None):
    """Return a tuple (conch_model, conch_preprocess, uni_model, uni_transform)
    HuggingFace token is loaded from HF_TOKEN environment variable.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    hf_token = os.getenv("HF_TOKEN", None)
    
    conch_model = None
    conch_preprocess = None
    uni_model = None
    uni_transform = None

    if "conch" in os.environ.get("DISABLE_MODELS", ""):
        logger.info("CONCH model disabled via DISABLE_MODELS env var")

    try:
        from conch.open_clip_custom import create_model_from_pretrained

        logger.info("Loading CONCH model...")
        pretrained_CONCH, preprocess_CONCH = create_model_from_pretrained(
            "conch_ViT-B-16",
            "hf_hub:MahmoodLab/conch",
            device=device,
            hf_auth_token=hf_token,
        )
        conch_model = pretrained_CONCH
        conch_preprocess = preprocess_CONCH
        logger.info("CONCH model loaded successfully")
    except Exception as e:
        logger.error(f"Could not load CONCH model: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load CONCH model: {e}") from e

    try:
        from uni import get_encoder
        from huggingface_hub import login

        logger.info("Loading UNI model...")
        if hf_token:
            login(token=hf_token)
        model_UNI, transform_UNI = get_encoder(enc_name="uni", device=device)
        uni_model = model_UNI
        uni_transform = transform_UNI
        logger.info("UNI model loaded successfully")
    except Exception as e:
        logger.error(f"Could not load UNI model: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load UNI model: {e}") from e

    return conch_model, conch_preprocess, uni_model, uni_transform


def extract_patch_embeddings(image: Image.Image, adata: anndata.AnnData, samplename: str, save_path: Path,
                             models: tuple, device: str = "cuda", save_augmented: bool = True, logger: logging.Logger = None):
    """Extract embeddings for a single sample and write them to disk.

    This function closely follows the logic in `dataset_preprocess.ipynb`.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    conch_model, conch_preprocess, uni_model, uni_transform = models

    def get_img_embd_conch(patch):
        base_width = 256
        patch_resized = patch.resize((base_width, base_width), Image.Resampling.LANCZOS)
        patch_processed = conch_preprocess(patch_resized).unsqueeze(0)
        with torch.inference_mode():
            feature_emb = conch_model.encode_image(patch_processed.to(device), proj_contrast=False, normalize=False)
        return torch.clone(feature_emb).detach().cpu()

    def get_img_embd_uni(patch):
        base_width = 224
        patch_resized = patch.resize((base_width, base_width), Image.Resampling.LANCZOS)
        img_transformed = uni_transform(patch_resized).unsqueeze(dim=0)
        with torch.inference_mode():
            feature_emb = uni_model(img_transformed.to(device))
        return torch.clone(feature_emb).detach().cpu()

    def patch_augmentation_embd(patch, conch_or_uni, num_transpose=7):
        if conch_or_uni == "conch":
            embd_dim = 512
        elif conch_or_uni == "uni":
            embd_dim = 1024
        else:
            raise ValueError("Unknown model for augmentation")
        patch_aug_embd = torch.zeros(num_transpose, embd_dim)
        for trans in range(num_transpose):
            patch_transposed = patch.transpose(trans)
            if conch_or_uni == "conch":
                patch_embd = get_img_embd_conch(patch_transposed)
            else:
                patch_embd = get_img_embd_uni(patch_transposed)
            patch_aug_embd[trans, :] = torch.clone(patch_embd)
        return patch_aug_embd.unsqueeze(0)

    # compute radius from adata
    spot_diameter = adata.uns["spatial"]["ST"]["scalefactors"]["spot_diameter_fullres"]
    if spot_diameter < 224:
        radius = 112
    else:
        radius = int(spot_diameter // 2)
    
    logger.debug(f"Sample {samplename}: spot_diameter={spot_diameter}, radius={radius}")

    x = adata.obsm["spatial"][:, 0]
    y = adata.obsm["spatial"][:, 1]
    num_spots = len(x)
    logger.info(f"Sample {samplename}: processing {num_spots} spots")

    all_patch_ebd_conch = None
    all_patch_ebd_uni = None
    all_patch_ebd_conch_aug = None
    all_patch_ebd_uni_aug = None

    first = True
    for spot_idx in tqdm(range(len(x)), desc=f"spots:{samplename}"):
        patch = image.crop((x[spot_idx] - radius, y[spot_idx] - radius, x[spot_idx] + radius, y[spot_idx] + radius))

        patch_ebd_conch = get_img_embd_conch(patch)
        patch_ebd_conch_aug = patch_augmentation_embd(patch, "conch")

        patch_ebd_uni = get_img_embd_uni(patch)
        patch_ebd_uni_aug = patch_augmentation_embd(patch, "uni")

        if first:
            all_patch_ebd_conch = patch_ebd_conch
            all_patch_ebd_uni = patch_ebd_uni
            all_patch_ebd_conch_aug = patch_ebd_conch_aug
            all_patch_ebd_uni_aug = patch_ebd_uni_aug
            first = False
        else:
            all_patch_ebd_conch = torch.cat((all_patch_ebd_conch, patch_ebd_conch), dim=0)
            all_patch_ebd_uni = torch.cat((all_patch_ebd_uni, patch_ebd_uni), dim=0)
            all_patch_ebd_conch_aug = torch.cat((all_patch_ebd_conch_aug, patch_ebd_conch_aug), dim=0)
            all_patch_ebd_uni_aug = torch.cat((all_patch_ebd_uni_aug, patch_ebd_uni_aug), dim=0)

    # save outputs
    save_path = Path(save_path)
    conch_path = save_path / f"processed_data/1spot_conch_ebd/{samplename}_conch.pt"
    uni_path = save_path / f"processed_data/1spot_uni_ebd/{samplename}_uni.pt"
    
    torch.save(all_patch_ebd_conch.detach().cpu(), conch_path)
    torch.save(all_patch_ebd_uni.detach().cpu(), uni_path)
    logger.info(f"Sample {samplename}: saved embeddings (CONCH: {all_patch_ebd_conch.shape}, UNI: {all_patch_ebd_uni.shape})")
    
    if save_augmented:
        conch_aug_path = save_path / f"processed_data/1spot_conch_ebd_aug/{samplename}_conch_aug.pt"
        uni_aug_path = save_path / f"processed_data/1spot_uni_ebd_aug/{samplename}_uni_aug.pt"
        torch.save(all_patch_ebd_conch_aug.detach().cpu(), conch_aug_path)
        torch.save(all_patch_ebd_uni_aug.detach().cpu(), uni_aug_path)
        logger.debug(f"Sample {samplename}: saved augmented embeddings")


def run(cfg: PreprocessConfig):
    save_path = cfg.save_path or cfg.data_path
    ensure_dirs(save_path)
    
    # Set up logging
    logger = setup_logging(save_path)
    logger.info("="*80)
    logger.info(f"Process HEST1K started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Configuration: selection_mode={cfg.selection_mode}")
    logger.info(f"Data path: {cfg.data_path}")
    logger.info(f"Save path: {save_path}")
    logger.info(f"Device: {cfg.device}")
    logger.info(f"Gene selection: {cfg.run_gene_selection}")
    
    ids = _resolve_ids(cfg, logger)
    logger.info(f"Total samples to process: {len(ids)}")
    if len(ids) == 0:
        logger.warning("No samples matched selection criteria. Exiting.")
        return

    models = None
    # Load image models only if image processing is enabled
    if cfg.process_mode in ("both", "images"):
        logger.info("Loading embedding models (image processing enabled)...")
        models = load_conch_and_uni(device=cfg.device, logger=logger)
    else:
        logger.info("Skipping embedding model loading (process_mode=%s)", cfg.process_mode)

    # Pre-check: validate that all required files exist
    logger.info("Pre-checking file availability...")
    tif_path = cfg.tif_path or (cfg.data_path / "wsis")
    st_path = cfg.st_path or (cfg.data_path / "st")
    
    missing_files = []
    for sid in ids:
        image_fp = tif_path / f"{sid}.tif"
        adata_fp = st_path / f"{sid}.h5ad"
        
        if not image_fp.exists():
            missing_files.append((sid, f"Image file: {image_fp}"))
        if not adata_fp.exists():
            missing_files.append((sid, f"ST file: {adata_fp}"))
    
    if missing_files:
        logger.error(f"Pre-check failed: {len(missing_files)} missing file(s):")
        for sid, missing in missing_files:
            logger.error(f"  - {sid}: {missing}")
        raise RuntimeError(f"Cannot proceed: {len(missing_files)} files not found. Fix paths in config and retry.")
    
    logger.info(f"Pre-check passed: all {len(ids)} samples have required files")
    processed = []
    failed = []

    # If user requested genes-only processing, skip image embedding extraction
    if cfg.process_mode == "genes":
        logger.info("Process mode set to 'genes' - skipping image embedding extraction."
                    " Will run gene selection on available ST files.")
        for sid in ids:
            adata_fp = (cfg.st_path or (cfg.data_path / "st")) / f"{sid}.h5ad"
            if not adata_fp.exists():
                logger.error(f"ST file not found: {adata_fp}; skipping {sid}")
                failed.append((sid, "ST file not found"))
                if cfg.halt_on_error:
                    logger.critical("halt_on_error=True: stopping execution after first error")
                    raise RuntimeError(f"Halted: ST file missing for {sid}")
                continue
            # For gene-only runs we consider the sample 'processed' (adata available)
            processed.append(sid)

    else:
        # images or both: perform image embedding extraction per sample
        for idx, sid in enumerate(ids, 1):
            logger.info(f"[{idx}/{len(ids)}] Processing {sid}...")

            tif_path = cfg.tif_path or (cfg.data_path / "wsis")
            st_path = cfg.st_path or (cfg.data_path / "st")

            image_fp = tif_path / f"{sid}.tif"
            adata_fp = st_path / f"{sid}.h5ad"

            if not image_fp.exists():
                logger.error(f"Image file not found: {image_fp}; skipping {sid}")
                failed.append((sid, "Image file not found"))
                if cfg.halt_on_error:
                    logger.critical("halt_on_error=True: stopping execution after first error")
                    raise RuntimeError(f"Halted: Image file missing for {sid}")
                continue
            if not adata_fp.exists():
                logger.error(f"ST file not found: {adata_fp}; skipping {sid}")
                failed.append((sid, "ST file not found"))
                if cfg.halt_on_error:
                    logger.critical("halt_on_error=True: stopping execution after first error")
                    raise RuntimeError(f"Halted: ST file missing for {sid}")
                continue

            try:
                image = Image.open(image_fp)
                adata = anndata.read_h5ad(adata_fp)
                # models must be available for image processing
                if models is None:
                    raise RuntimeError("Image models not loaded but image processing requested")
                extract_patch_embeddings(image, adata, sid, save_path, models, device=cfg.device,
                                         save_augmented=cfg.save_augmented, logger=logger)
                processed.append(sid)
                logger.info(f"[{idx}/{len(ids)}] {sid} completed successfully")
            except Exception as e:
                logger.error(f"Error processing {sid}: {e}", exc_info=True)
                failed.append((sid, str(e)))
                if cfg.halt_on_error:
                    logger.critical("halt_on_error=True: stopping execution after first error")
                    raise RuntimeError(f"Halted after error on {sid}: {e}") from e
    
    
    # optional: gene selection across processed slides
    # Only run gene selection when the process_mode allows it (both or genes)
    if cfg.run_gene_selection and processed and cfg.process_mode in ("both", "genes"):
        logger.info(f"Running gene selection on {len(processed)} processed samples...")

        # First: compute common genes across all processed slides (keep only intersecting genes)
        logger.info("Computing intersection of genes across processed slides (common_genes)")
        common_genes = None
        for sid in processed:
            adata = anndata.read_h5ad((cfg.st_path or (cfg.data_path / "st")) / f"{sid}.h5ad")
            genes = set(adata.var_names)
            if common_genes is None:
                common_genes = genes
            else:
                common_genes = common_genes.intersection(genes)
        if common_genes is None or len(common_genes) == 0:
            logger.error("No common genes found across processed slides. Cannot run gene selection.")
            raise RuntimeError("No common genes across slides")
        common_genes = sorted(list(common_genes))
        logger.info(f"Length of common genes: {len(common_genes)}")

        # Now compute union of HVGs on the restricted common_genes set
        union_hvg = set()
        sc = __import__("scanpy")
        for sid in processed:
            logger.debug(f"Computing HVGs for {sid} using common_genes subset")
            adata = anndata.read_h5ad((cfg.st_path or (cfg.data_path / "st")) / f"{sid}.h5ad")
            # reduce to common genes before HVG computation
            ad = adata[:, common_genes].copy()
            sc.pp.filter_cells(ad, min_genes=1)
            sc.pp.filter_genes(ad, min_cells=1)
            sc.pp.normalize_total(ad, inplace=True)
            sc.pp.log1p(ad)
            sc.pp.highly_variable_genes(ad, n_top_genes=2000)   # using 2000 as in the original notebook
            hvg_count = int(ad.var["highly_variable"].sum())
            union_hvg = union_hvg.union(set(ad.var_names[ad.var["highly_variable"]]))
            logger.debug(f"{sid}: {hvg_count} highly-variable genes; union total: {len(union_hvg)}")

        union_hvg = sorted(list(union_hvg))
        logger.info(f"Union HVG set size (within common_genes): {len(union_hvg)}")

        # order and select final genes similar to notebook, using only common_genes/union_hvg
        all_count_df = pd.DataFrame()
        for sid in processed:
            adata = anndata.read_h5ad((cfg.st_path or (cfg.data_path / "st")) / f"{sid}.h5ad")
            # restrict to union_hvg which is itself a subset of common_genes
            df = pd.DataFrame(adata[:, union_hvg].X.toarray(), columns=union_hvg)
            all_count_df = pd.concat([all_count_df, df], axis=0) if not all_count_df.empty else df

        all_count_df.fillna(0, inplace=True)
        mean_order = all_count_df.mean(axis=0).sort_values(ascending=False).index
        std_order = all_count_df.std(axis=0).sort_values(ascending=False).index
        # HVG list: union HVGs ordered by mean expression (descending)
        if cfg.hvg_top_k > 0:
            hvg_genes = list(mean_order[: min(cfg.hvg_top_k, len(mean_order))])
            hvg_out_fn = (
                save_path / "processed_data" / _append_suffix_to_filename(cfg.gene_list_filename, "_hvg")
            )
            hvg_out_fn.parent.mkdir(parents=True, exist_ok=True)
            with hvg_out_fn.open("w") as f:
                for g in hvg_genes:
                    f.write(g + "\n")
            logger.info(f"Saved {len(hvg_genes)} HVG genes (mean-ordered) to {hvg_out_fn}")
        else:
            logger.info("HVG selection skipped (hvg_top_k=0)")

        # HMHVG list: grow window one-by-one from hmhvg_top_k until the intersection is large enough
        if cfg.hmhvg_top_k > 0:
            max_len = min(len(mean_order), len(std_order))
            window = min(cfg.hmhvg_top_k, max_len)
            hmhvg_candidates = set()
            while window <= max_len:
                top_mean_set = set(mean_order[:window])
                top_std_set = set(std_order[:window])
                hmhvg_candidates = top_mean_set.intersection(top_std_set)
                if len(hmhvg_candidates) >= cfg.hmhvg_top_k or window == max_len:
                    break
                window += 1

            hmhvg_genes = sorted(list(hmhvg_candidates))[: min(cfg.hmhvg_top_k, len(hmhvg_candidates))]
            if len(hmhvg_genes) < cfg.hmhvg_top_k:
                logger.warning(
                    f"Requested {cfg.hmhvg_top_k} HMHVG genes but only found {len(hmhvg_genes)} using window={window};"
                    " consider increasing hmhvg_top_k if you need more."
                )
            hmhvg_out_fn = (
                save_path / "processed_data" / _append_suffix_to_filename(cfg.gene_list_filename, "_hmhvg")
            )
            hmhvg_out_fn.parent.mkdir(parents=True, exist_ok=True)
            with hmhvg_out_fn.open("w") as f:
                for g in hmhvg_genes:
                    f.write(g + "\n")
            logger.info(f"Saved {len(hmhvg_genes)} HMHVG genes (high-mean/high-std) to {hmhvg_out_fn}")
        else:
            logger.info("HMHVG selection skipped (hmhvg_top_k=0)")

        # DEG list: union of per-group DEGs across slides
        if cfg.deg_top_k > 0:
            meta_df = _load_metadata_for_deg(cfg, logger)
            if not cfg.deg_groupby:
                cfg.deg_groupby = "region"
                logger.info("DEG: deg_groupby not set; defaulting to 'region'")

            deg_union = []
            for sid in processed:
                logger.debug(f"Computing DEGs for {sid} grouped by '{cfg.deg_groupby}'")
                adata = anndata.read_h5ad((cfg.st_path or (cfg.data_path / "st")) / f"{sid}.h5ad")
                if cfg.deg_groupby not in adata.obs:
                    label_path = _get_label_path_for_id(sid, cfg, meta_df)
                    if label_path is None or not label_path.exists():
                        logger.info(
                            f"DEG: no label file for {sid} (expected {label_path}); skipping slide for DEG"
                        )
                        continue
                    adata = _inject_region_labels_from_subseries(adata, sid, cfg, meta_df, logger, label_path=label_path)
                if cfg.deg_groupby not in adata.obs:
                    logger.warning(
                        f"DEG: column '{cfg.deg_groupby}' not found in obs for {sid} and injection failed; skipping slide for DEG"
                    )
                    continue
                # drop unlabeled spots before DEG
                labeled_mask = ~adata.obs[cfg.deg_groupby].isna()
                if labeled_mask.sum() == 0:
                    logger.warning(f"DEG: no labeled spots for {sid} after injection; skipping slide for DEG")
                    continue
                ad = adata[labeled_mask, common_genes].copy()
                sc.pp.filter_cells(ad, min_genes=1)
                sc.pp.filter_genes(ad, min_cells=1)
                sc.pp.normalize_total(ad, inplace=True)
                sc.pp.log1p(ad)
                groups_arg = cfg.deg_groups if cfg.deg_groups else ad.obs[cfg.deg_groupby].unique().tolist()
                if len(groups_arg) == 0:
                    logger.warning(f"DEG: no groups found for {sid}; skipping slide for DEG")
                    continue
                sc.tl.rank_genes_groups(
                    ad,
                    groupby=cfg.deg_groupby,
                    n_genes=cfg.deg_top_k,
                    groups=groups_arg,
                )
                deg_names = ad.uns["rank_genes_groups"]["names"]
                # deg_names can be dict-like or ndarray/recarray; normalize to columns
                try:
                    columns = deg_names.T
                except Exception:
                    # fallback for dict-like
                    if isinstance(deg_names, dict):
                        columns = deg_names.values()
                    else:
                        columns = []
                for col in columns:
                    try:
                        deg_union.extend(list(col[: cfg.deg_top_k]))
                    except Exception:
                        # some scanpy versions return a 0-d object; try flattening
                        try:
                            deg_union.extend(list(col)[0: cfg.deg_top_k])
                        except Exception:
                            logger.warning("DEG: could not parse rank_genes_groups names column; skipping a column")
                logger.debug(f"{sid}: DEG collected genes count now {len(deg_union)}")

            # Final cap: deduplicate and cap to deg_top_k
            deg_genes = sorted(list(dict.fromkeys(deg_union)))[: cfg.deg_top_k]
            deg_out_fn = (
                save_path / "processed_data" / _append_suffix_to_filename(cfg.gene_list_filename, "_deg")
            )
            deg_out_fn.parent.mkdir(parents=True, exist_ok=True)
            with deg_out_fn.open("w") as f:
                for g in deg_genes:
                    f.write(g + "\n")
            logger.info(f"Saved {len(deg_genes)} DEG genes (union across groups/slides) to {deg_out_fn}")
        else:
            logger.info("DEG selection skipped (deg_top_k=0)")
    
    # Save all_slide_lst.txt with successfully processed sample IDs
    if processed:
        all_slide_lst_fn = save_path / "processed_data/all_slide_lst.txt"
        all_slide_lst_fn.parent.mkdir(parents=True, exist_ok=True)
        with all_slide_lst_fn.open("w") as f:
            for sid in processed:
                f.write(sid + "\n")
        logger.info(f"Saved {len(processed)} processed sample IDs to {all_slide_lst_fn}")
    
    # final summary
    logger.info("="*80)
    logger.info(f"Processing complete. Successfully processed: {len(processed)}/{len(ids)}")
    if failed:
        logger.warning(f"Failed samples ({len(failed)}):")
        for sid, reason in failed:
            logger.warning(f"  - {sid}: {reason}")
    logger.info(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log saved to: {save_path}/processed_data/process_hest1k.log")


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Preprocess HEST1K samples via TOML config")
    p.add_argument("-c", "--config", required=True, type=Path, help="Path to TOML config")
    return p.parse_args().config


def _cli_entrypoint():
    cfg_path = parse_args()
    try:
        cfg: PreprocessConfig = load_toml_config(cfg_path, PreprocessConfig, sections_to_flatten=["paths"])
        print(f"Loaded config from: {cfg_path}")
        run(cfg)
    except Exception as e:
        print(f"Fatal error: {e}", file=__import__("sys").stderr)
        raise


if __name__ == "__main__":
    _cli_entrypoint()
