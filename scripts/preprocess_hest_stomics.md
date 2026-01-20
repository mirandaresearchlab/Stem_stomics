# preprocess_hest_stomics: preprocessing pipeline

This script normalizes gene names, groups samples by gene panel, filters genes, saves per-sample processed AnnData files, and builds text embeddings for downstream use. It is meant to be run as a CLI:

```
python scripts/preprocess_hest_stomics.py \
  --metadata /storage/hest1k/HEST_v1_1_0.csv \
  --st-path /storage/hest1k/st \
  --log-file logs/preprocess_hest_stomics.log
```

## Inputs
- Metadata CSV (`--metadata`), expected to contain at least: `id`, `dataset_title`, `species`, `st_technology`, `organ`, `preservation_method`.
- Directory with ST `.h5ad` files (`--st-path`), one per `id`.

## Steps (in order)
1) **Logging**: writes to stdout and `--log-file`.
2) **Metadata slice**: keep rows where `species == "Homo sapiens"` and `st_technology != "Spatial Transcriptomics"`.
3) **Load `.h5ad` files**:
   - Drop duplicate genes per file (keep first occurrence).
   - Collect any Ensembl IDs (ENSG*).
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
   - Write each processed slide to `st_processed_<YYYYMMDD>/<id>.h5ad` with `dataset_title` reset to the base title.
10) **Stats**:
    - Per-spot min/max/mean across genes (describe output) per split.
    - Random slide value summary per split.
11) **Embedding cache (in memory)**:
    - Collect unique strings from metadata: `organ`, `st_technology`, `preservation_method`, plus all surviving genes.
    - Encode with `thomas-sounack/BioClinical-ModernBERT-base` (SentenceTransformer); log if embeddings are unique.

## Outputs
- Processed `.h5ad` files in `st_processed_<YYYYMMDD>/`.
- Logs summarizing:
  - Duplicate gene drops.
  - Ensembl translation stats.
  - Split counts per dataset.
  - Gene drops per split and surviving gene count.
  - Dropped splits (>90% genes removed).
  - Per-spot stats and random-slide summaries.
  - Embedding build summary (count, model).

## Notes and assumptions
- Panel grouping is order-insensitive; same genes in different order will still be grouped.
- If mygene returns no translations, Ensembl genes are dropped and the run continues.
- Constant-gene filtering is per-split; splits with empty panels or extreme drops are skipped.
- Embeddings are held in memory only (not persisted). Save them if you need reuse.***
