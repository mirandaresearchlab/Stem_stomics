Process HEST1K samples (script)
================================

What this script does
---------------------
- Loads ST `h5ad` and matching `*.tif` slide images from an existing HEST1K dataset.
- Extracts per-spot image patch embeddings using the CONCH and UNI encoders (same approach as
  in `dataset_preprocess.ipynb`).
- Saves embeddings under `processed_data/` inside the configured `save_path` (or `data_path`).
- Optionally computes and saves gene lists across processed slides: HVG (mean-ordered highly-variable genes),
  HMHVG (high-mean/high-std genes), and DEG (union of per-group DEGs). Lists share the same base filename with
  `_hvg` / `_hmhvg` / `_deg` suffixes (default base: `processed_data/selected_gene_list.txt`). For HER2ST, DEG labels
  can be auto-injected from `deg_label_dir` (e.g., `data/ST-pat/lbl/<subseries>_labeled_coordinates.tsv`) using
  the HEST metadata `subseries` column.

Key points / compatibility notes
------------------------------
- The script expects the dataset to already be downloaded. It will not attempt to download anything.
- Configuration is read from a TOML file and validated using the project's `utils/config_loader.load_toml_config`.
- The script reads the HuggingFace authentication token only from the environment variable `HF_TOKEN`. Do not put your token in the TOML file.
- Model loading is strict: if a required model (CONCH / UNI) fails to load the script will raise an error (fail-fast). There are no silent zero-tensor fallbacks.
- The script accepts a `process_mode` option in the TOML which controls what is done: `both` (default), `images`, or `genes`.
- The `gene_list_filename` option (default: `selected_gene_list.txt`) lets you customize the output filename for selected genes,
  allowing multiple gene-selection runs to save different outputs without overwriting.
- The script aims to keep the same folder layout as the notebook: `processed_data/1spot_conch_ebd/`,
  `processed_data/1spot_uni_ebd/`, and their `_aug` counterparts.

Minimal TOML example
---------------------
Create a TOML file like this and pass it with `-c config.toml`:

```toml
[paths]
data_path = "/storage/hest1k"
# optional overrides; if not set, script uses data_path/wsis and data_path/st
tif_path = "/storage/hest1k/wsis"
st_path = "/storage/hest1k/st"

ids_to_query = ["MEND65", "MISC3", "NCBI631"]

device = "cuda"
# optional: hf_token can be set here or via HF_TOKEN env var
hf_token = ""

# optional: run gene selection after embeddings
run_gene_selection = true
hvg_top_k = 2000   # set 0 to skip HVG output
hmhvg_top_k = 200  # set 0 to skip HMHVG output
deg_top_k = 0      # set >0 to enable DEG output; requires deg_groupby
deg_groupby = ""   # e.g., "region" or other obs column with labels; if empty, defaults to 'region' and tries label injection
deg_label_dir = "data/ST-pat/lbl"  # optional: per-slide region label files named <subseries>_labeled_coordinates.tsv
# results will be written to `selected_gene_list_hvg.txt`, `_hmhvg.txt`, and `_deg.txt` (as enabled)
```

How to run
----------
From repository root:

```bash
python scripts/process_hest1k.py -c path/to/config.toml
```

Notes on differences from notebooks
----------------------------------
- Functions were adapted from `dataset_preprocess.ipynb` and structured so the script can be
  invoked from CLI using a TOML config.
- The script centralises model loading for efficiency (models are loaded once per run).
- The script adds basic checks for missing files and prints informative messages.
