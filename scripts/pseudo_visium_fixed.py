# NOTE: This code is adapted from https://github.com/mahmoodlab/HEST (HEST/src/hest/readers.py, HEST/src/hest/HESTData.py).
# Changes vs upstream: (1) use safe dtypes (int64) to avoid uint16 overflow when pooling; (2) handle sparse inputs explicitly;
# (3) filter zero-count bins consistently; (4) pad grid extents and clip indices to avoid edge drops; (5) optional patch dumping
# filters AnnData alongside coordinates so the counts and patches stay aligned.

import math
from pathlib import Path
from typing import Literal, Union, Optional, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from PIL import Image


def pool_bins_visiumhd_fixed(
    adata: sc.AnnData,
    pixel_size: float,
    dst_bin_size_um: int = 128,
    src_bin_size_um: Literal[2, 8, 16] = 16,
    chunk_len: int = 50_000,
) -> sc.AnnData:
    """Pool a Visium HD AnnData (src_bin_size_um) into pseudo-Visium bins (dst_bin_size_um).

    Fixes vs upstream:
    - Accepts sparse X by densifying per chunk.
    - Accumulates in int64 to avoid overflow.
    - Pads grid extents and clips indices to avoid dropping edge bins.
    - Filters zero-count bins before returning.
    """
    if src_bin_size_um >= dst_bin_size_um:
        raise ValueError("dst_bin_size_um needs to be larger than src_bin_size_um")
    if dst_bin_size_um % src_bin_size_um != 0:
        raise ValueError("dst_bin_size_um must be a multiple of src_bin_size_um")

    y_max = adata.obs["pxl_row_in_fullres"].max()
    y_min = adata.obs["pxl_row_in_fullres"].min()
    x_max = adata.obs["pxl_col_in_fullres"].max()
    x_min = adata.obs["pxl_col_in_fullres"].min()

    dst_bin_pxl_size = dst_bin_size_um / pixel_size
    src_bin_pxl_size = src_bin_size_um / pixel_size

    # pad by one dst bin to avoid edge drop
    grid_height_pxl = (y_max - y_min) + dst_bin_pxl_size
    grid_width_pxl = (x_max - x_min) + dst_bin_pxl_size

    m = math.ceil(grid_height_pxl / dst_bin_pxl_size)
    n = math.ceil(grid_width_pxl / dst_bin_pxl_size)

    features = adata.var_names
    spot_grid = np.zeros((m * n, len(features)), dtype=np.int64)

    a = np.floor((adata.obs["pxl_col_in_fullres"] - x_min + src_bin_pxl_size / 2) / dst_bin_pxl_size).astype(int)
    b = np.floor((adata.obs["pxl_row_in_fullres"] - y_min + src_bin_pxl_size / 2) / dst_bin_pxl_size).astype(int)
    # clip to grid
    a = np.clip(a, 0, n - 1)
    b = np.clip(b, 0, m - 1)
    c = (b * n + a).to_numpy()

    nb_chunks = int(np.ceil(len(c) / chunk_len))
    X = adata.X
    for i in range(nb_chunks):
        start, end = i * chunk_len, min((i + 1) * chunk_len, len(c))
        chunk_indices = c[start:end]
        chunk_X = X[start:end]
        if sparse.issparse(chunk_X):
            chunk_X = chunk_X.toarray()
        spot_grid[chunk_indices] += np.asarray(chunk_X, dtype=np.int64)

    expression_df = pd.DataFrame(spot_grid, columns=features)
    row_sums = expression_df.sum(axis=1)
    expression_df = expression_df[row_sums > 0]

    pos_x = x_min + (expression_df.index % n) * dst_bin_pxl_size + dst_bin_pxl_size / 2
    pos_y = y_min + np.floor(expression_df.index / n) * dst_bin_pxl_size + dst_bin_pxl_size / 2

    pooled = sc.AnnData(expression_df)
    pooled.obsm["spatial"] = np.column_stack((pos_x, pos_y))
    pooled.obs["in_tissue"] = True
    pooled.obs["pxl_col_in_fullres"] = pos_x
    pooled.obs["pxl_row_in_fullres"] = pos_y
    pooled.obs["array_col"] = np.arange(len(pooled.obs)) % n
    pooled.obs["array_row"] = np.arange(len(pooled.obs)) // n
    pooled.obs.index = [f"{r:04d}x{c:04d}" for r, c in zip(pooled.obs["array_row"], pooled.obs["array_col"])]
    pooled.uns["pixel_size"] = pixel_size
    pooled.uns["spot_diameter_um"] = dst_bin_size_um
    pooled.uns["spot_diameter_fullres"] = dst_bin_size_um / pixel_size
    return pooled


def pool_transcripts_xenium_fixed(
    df: Union[pd.DataFrame, "dd.DataFrame"],  # type: ignore
    pixel_size_he: float,
    spot_size_um: float = 100.0,
    key_x: str = "he_x",
    key_y: str = "he_y",
) -> sc.AnnData:
    """Pool Xenium transcripts into square bins of `spot_size_um`.

    Fixes vs upstream:
    - Accumulates in int64 to avoid uint16 overflow.
    - Pads grid extents and clips indices to avoid edge drops.
    - Filters zero-count bins before returning.
    """
    import dask.dataframe as dd  # local import to avoid hard dep if unused

    y_max = df[key_y].max()
    y_min = df[key_y].min()
    x_max = df[key_x].max()
    x_min = df[key_x].min()

    if isinstance(df, dd.DataFrame):
        y_max, y_min, x_max, x_min = dd.compute(y_max, y_min, x_max, x_min)

    span_y = (y_max - y_min) + (spot_size_um / pixel_size_he)
    span_x = (x_max - x_min) + (spot_size_um / pixel_size_he)
    m = math.ceil(span_y / (spot_size_um / pixel_size_he))
    n = math.ceil(span_x / (spot_size_um / pixel_size_he))

    features = df["feature_name"].unique()
    if isinstance(df, dd.DataFrame):
        features = features.compute()

    spot_grid = np.zeros((m * n, len(features)), dtype=np.int64)
    a = np.floor((df[key_x] - x_min) / (spot_size_um / pixel_size_he)).astype(int)
    b = np.floor((df[key_y] - y_min) / (spot_size_um / pixel_size_he)).astype(int)
    if isinstance(df, dd.DataFrame):
        a, b = dd.compute(a, b)
    a = np.clip(np.asarray(a), 0, n - 1)
    b = np.clip(np.asarray(b), 0, m - 1)
    c = b * n + a

    # resolve columns for feature indices
    if isinstance(df, dd.DataFrame):
        cols = pd.Index(features).get_indexer(df["feature_name"].compute())
    else:
        cols = pd.Index(features).get_indexer(df["feature_name"])
    np.add.at(spot_grid, (c, cols), 1)

    expression_df = pd.DataFrame(spot_grid, columns=features)
    row_sums = expression_df.sum(axis=1)
    expression_df = expression_df[row_sums > 0]

    pos_x = x_min + (expression_df.index % n) * (spot_size_um / pixel_size_he) + (spot_size_um / (2 * pixel_size_he))
    pos_y = y_min + np.floor(expression_df.index / n) * (spot_size_um / pixel_size_he) + (spot_size_um / (2 * pixel_size_he))

    pooled = sc.AnnData(expression_df)
    pooled.obsm["spatial"] = np.column_stack((pos_x, pos_y))
    pooled.obs["in_tissue"] = True
    pooled.obs["pxl_col_in_fullres"] = pos_x
    pooled.obs["pxl_row_in_fullres"] = pos_y
    pooled.obs["array_col"] = np.arange(len(pooled.obs)) % n
    pooled.obs["array_row"] = np.arange(len(pooled.obs)) // n
    pooled.obs.index = [f"{r:04d}x{c:04d}" for r, c in zip(pooled.obs["array_row"], pooled.obs["array_col"])]
    return pooled


def dump_patches_fixed(
    adata: sc.AnnData,
    wsi_path: Union[str, Path],
    patch_save_dir: Union[str, Path],
    name: str = "patches",
    target_patch_size: int = 224,
    target_pixel_size: float = 0.5,
    use_mask: bool = True,
    tissue_mask: Optional[np.ndarray] = None,
):
    """Dump image patches centered on adata.obsm['spatial'] coordinates.

    Fixes vs upstream: filters adata and coords together when spots fall outside the WSI to keep alignment.
    """
    Image.MAX_IMAGE_PIXELS = None  # avoid PIL decompression bomb warnings on large WSIs
    patch_save_dir = Path(patch_save_dir)
    patch_save_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(wsi_path).convert("RGB")
    src_pixel_size = getattr(adata, "uns", {}).get("pixel_size", None)
    if src_pixel_size is None:
        src_pixel_size = target_pixel_size  # assume already at target scale if unknown

    coords_center = np.asarray(adata.obsm["spatial"])
    # compute crop size in source pixels to cover target FOV at target_pixel_size
    patch_fov_um = target_patch_size * target_pixel_size
    patch_size_src = patch_fov_um / src_pixel_size  # in source pixels
    coords_topleft = coords_center - patch_size_src / 2.0

    w, h = img.size
    in_slide_mask = (
        (coords_topleft[:, 0] >= 0)
        & (coords_topleft[:, 1] >= 0)
        & (coords_topleft[:, 0] + patch_size_src <= w)
        & (coords_topleft[:, 1] + patch_size_src <= h)
    )

    coords_center = coords_center[in_slide_mask]
    adata = adata[in_slide_mask].copy()

    patches = []
    for cx, cy in coords_center:
        left = int(cx - patch_size_src / 2.0)
        upper = int(cy - patch_size_src / 2.0)
        right = int(left + patch_size_src)
        lower = int(upper + patch_size_src)
        patch = img.crop((left, upper, right, lower))
        # resample to target pixel size (target_patch_size at target_pixel_size)
        patch = patch.resize((target_patch_size, target_patch_size), resample=Image.BILINEAR)
        patches.append(np.asarray(patch))

    np.savez_compressed(patch_save_dir / f"{name}_images.npz", images=np.stack(patches), index=adata.obs_names.to_numpy())
    adata.write(patch_save_dir / f"{name}_adata.h5ad")
    return adata, patches
