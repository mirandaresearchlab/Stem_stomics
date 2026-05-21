"""
Spot-mean *oracle* baseline.

For each spot s in the held-out slide, predict the same scalar
m_s = (1/C) * sum_i log2(X[s, i] + 1)
for every gene i. This is an oracle — it uses the test-slide ground truth
to compute m_s — and exists to reproduce the PCC-trap observation of
Zhu et al. (2025): "PCC in log-transformed space would be surprisingly high
if the prediction is simply the mean expression across all genes in this spot."

It is NOT a usable inference-time predictor; report it as `[oracle]` next to
real models to expose PCC as a gameable metric.

Output is a tensor of shape (N_spots * num_rep, 1, N_genes), matching the
format produced by scripts/stem_sample.py / scripts/fm_sample.py, so
scripts/eval.py consumes it unchanged.
"""

import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch

from settings.spot_mean_oracle import SpotMeanOracleConfig
from utils.config_loader import load_toml_config


def _read_st_h5ad(data_path: Path, st_subdir: str, slide: str) -> anndata.AnnData:
    candidates = [
        data_path / st_subdir / f"{slide}.h5ad",
        data_path / ".." / st_subdir / f"{slide}.h5ad",
    ]
    for p in candidates:
        if p.exists():
            return anndata.read_h5ad(p)
    raise FileNotFoundError(
        f"Could not find h5ad for slide '{slide}'. Looked in: {[str(p) for p in candidates]}"
    )


def _slide_count_mtx(adata: anndata.AnnData, selected_genes: list[str]) -> pd.DataFrame:
    X = adata[:, selected_genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return pd.DataFrame(X, columns=selected_genes, index=adata.obs_names)


def main(cfg: SpotMeanOracleConfig) -> None:
    processed_root = cfg.data_path / cfg.processed_subdir
    selected_genes = list(np.genfromtxt(processed_root / cfg.gene_list_filename, dtype=str))
    n_genes = len(selected_genes)
    print(f"Selected genes: {n_genes}  (file={cfg.gene_list_filename})")

    test_adata = _read_st_h5ad(cfg.data_path, cfg.st_subdir, cfg.slide_out)
    cm = _slide_count_mtx(test_adata, selected_genes)
    n_spots = cm.shape[0]
    print(f"Held-out slide '{cfg.slide_out}': {n_spots} spots")

    log_cm = np.log2(cm.values + 1).astype(np.float32)               # (n_spots, n_genes)
    per_spot_mean = log_cm.mean(axis=1, keepdims=True)               # (n_spots, 1)
    pred = np.broadcast_to(per_spot_mean, (n_spots, n_genes)).copy() # (n_spots, n_genes)

    t = torch.from_numpy(pred)                                       # (n_spots, n_genes)
    t = t.view(n_spots, 1, n_genes).expand(-1, cfg.num_rep, -1)      # (n_spots, num_rep, n_genes)
    t = t.reshape(n_spots * cfg.num_rep, 1, n_genes).contiguous()    # eval.py layout

    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.save_dir / f"spot_mean_oracle_{cfg.slide_out}_{cfg.num_rep}sample.pt"
    torch.save(t, out_path)
    print(f"Saved oracle samples → {out_path}  shape={tuple(t.shape)}")
    print(
        f"Per-spot scalar stats: min={per_spot_mean.min():.4f} "
        f"max={per_spot_mean.max():.4f} mean={per_spot_mean.mean():.4f} "
        f"std={per_spot_mean.std():.4f}"
    )


def parse_args() -> Path:
    parser = argparse.ArgumentParser(
        description="Generate the spot-mean *oracle* baseline predictions (uses test GT)."
    )
    parser.add_argument(
        "-c", "--config", required=True, type=Path, metavar="FILE",
        help="Path to oracle TOML config (required)",
    )
    return parser.parse_args().config


def _cli_entrypoint() -> None:
    cfg_path = parse_args()
    cfg: SpotMeanOracleConfig = load_toml_config(
        cfg_path, SpotMeanOracleConfig, sections_to_flatten=["data", "baseline"]
    )
    print("▶ loaded spot-mean oracle config:\n", cfg)
    main(cfg)


if __name__ == "__main__":
    _cli_entrypoint()
