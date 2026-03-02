# preprocess_hest_stomics: preprocessing pipeline

This script normalizes gene names, groups samples by gene panel, filters genes, saves per-sample processed AnnData files, generates patches, and builds text embeddings for downstream use. It is meant to be run as a CLI:

```
python scripts/preprocess_hest_stomics.py \
  --metadata /storage/hest1k/HEST_v1_2_1.csv \
  --st-path /storage/hest1k/st \
  --log-file logs/preprocess_hest_stomics.log
```

## Inputs
- Metadata CSV (`--metadata`) with `id`, `dataset_title`, `species`, `st_technology`, `organ`, `preservation_method`, `pixel_size_um_estimated` (required), `pixel_size_um_embedded`, `spot_diameter`.
- Directory with ST `.h5ad` files (`--st-path`), one per `id`.
- Root directory with slide `.tif/.tiff` WSIs (`--tif-root`), expected as `<id>.tif`. Used for patch extraction.

## Pipe
1) **Logging**: writes to stdout and `--log-file`.
2) **Metadata slice**: keep rows where `species == "Homo sapiens"` and `st_technology != "Spatial Transcriptomics"`. Log datasets per organ after filtering.
3) **Load `.h5ad` files**:
   - Drop duplicate genes per file (keep first occurrence).
   - Collect any Ensembl IDs (ENSG*).
   - Enforce `pixel_size_um_estimated` (must exist in `adata.uns` or metadata; otherwise error) and store it in `adata.uns`.
   - Attach metadata fields (organ, st_technology, preservation_method, pixel size, spot diameter) to `obs`.
   - **Visium HD pooling**: if spatial cols exist, pool 16 µm bins into **96 µm** pseudo-spots (6×16 µm; contiguous tiling) using `pool_bins_visiumhd_fixed`.
     - Rationale: preserves divisibility by the 16 µm base grid while targeting the default 96 µm pseudo-spot size.
     - Edge bins outside the grid are dropped (upstream-style behavior).
   - **Xenium**: treated as already spot-level; no pooling here (packaged h5ads already on the intended lattice with ~100 µm spacing per HEST Issue [#79](https://github.com/mahmoodlab/HEST/issues/79)). `spot_size_um` is set to 100 µm.
4) **Ensembl translation**:
   - Async lookup via mygene; translation failures log warnings.
   - Untranslated Ensembl genes are dropped during normalization.
5) **Gene normalization**:
   - For each AnnData: translate Ensembl IDs when possible; lowercase; strip `grch38_` prefixes; drop duplicates; drop untranslatable Ensembl IDs.
6) **Panel grouping**:
   - Group slides by gene panel (order-insensitive: sorted var_names).
   - If multiple panels exist in a dataset, create splits named `(Split X) <dataset_title>`; otherwise keep original title.
7) **Common genes per split**:
   - Slides within a split already share the same panel; intersect implicitly by using that panel.
8) **Constant-gene filtering**:
   - Concatenate split slides; drop genes whose max == min across all spots.
   - If >90% of genes would be dropped, skip the split and record it as dropped.
9) **Save processed data**:
   - Keep only non-constant genes per slide.
   - Write each processed slide to `processed_<YYYYMMDD>/h5ad/<id>.h5ad` with `dataset_title` reset to the base title; `spot_id` prefixed with slide id.
10) **Patches**:
    - If `<id>.tif` exists, extract **224×224** patches at **0.5 µm/px** (~112 µm FOV) centered on spots, resampling from source pixel size; save to `processed_<date>/patches/slide=<id>_images.npz` and write corresponding filtered AnnData.
    - Out-of-bounds spots are logged to `processed_<date>/patch_oob.log`. Patch extraction failures stop the run.
    - For slides with patches, out-of-bounds spots are excluded from `spots.parquet` to keep `spot_id` fully aligned with patch indices.
11) **Stats**:
    - Per-gene min/q1/median/q3/max/mean across spots per split.
    - Per-spot min/max/mean across genes per split.
    - Random slide value summary per split.
    - **Sanity checks**: nearest-neighbor spacing logged per slide; expects ~100 µm (Visium/Xenium) or ~96 µm (Visium HD); warns on >15% deviation.
12) **Embeddings (persisted)**:
    - Collect unique strings from metadata: `organ`, `st_technology`, `preservation_method`, plus all surviving genes.
    - Encode with `thomas-sounack/BioClinical-ModernBERT-base`; save `embeddings.npy`, mappings, and meta under `processed_<date>/embeddings/`.
13) **Spots metadata**:
    - Save `processed_<date>/spots.parquet` with one row per recorded spot: `spot_id`, `slide_id`, `dataset_title`, `organ`, `st_technology`, `preservation_method`, spatial coords (px), `pixel_size_um`, `h5ad_file`, `patch_file`, `tif_path`.

## Outputs
- Processed `.h5ad` files in `processed_<YYYYMMDD>/h5ad/`.
- Patches in `processed_<YYYYMMDD>/patches/slide=<id>_images.npz` (+ filtered adata alongside).
- Text embeddings in `processed_<YYYYMMDD>/embeddings/`.
- Spot metadata in `processed_<YYYYMMDD>/spots.parquet`.
- Out-of-bounds patch report in `processed_<YYYYMMDD>/patch_oob.log`.
- Processed AnnData includes `spot_size_um` for Visium HD (96 µm default) and Xenium (100 µm default).
- Logs summarizing:
  - Duplicate gene drops.
  - Ensembl translation stats.
  - Split counts per dataset.
  - Datasets per organ (post-filter).
  - Gene drops per split and surviving gene count.
  - Dropped splits (>90% genes removed).
  - Per-gene/spot stats and random-slide summaries.
  - NN spacing sanity checks vs expected scales.
  - Patch resampling info (FOV, ratio).
  - Embedding build summary (count, model).

## Notes and assumptions
- Panel grouping is order-insensitive; same genes in different order are grouped.
- If mygene returns no translations, Ensembl genes are dropped and the run continues.
- Constant-gene filtering is per-split; splits with empty panels or extreme drops are skipped.
- Visium HD pseudo-spots target 96 µm; Xenium in packaged h5ads is treated as already on its intended lattice (expected ~100 µm spacing per Issue #79); native Visium retains its ~55 µm diameter, ~100 µm spacing.
- Patches are resampled to 0.5 µm/px (224×224); `pixel_size_um_estimated` is required for all samples.
