"""
Ported from eval.ipynb to a script with TOML config.
"""

import argparse
import os
from pathlib import Path

import anndata
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from settings.eval import EvalConfig
from utils.config_loader import load_toml_config


def load_data(cfg: EvalConfig):
    processed_root = cfg.data_path / cfg.processed_subdir
    st_root = cfg.data_path / ".." / cfg.st_subdir

    slidename_lst = list(np.genfromtxt(processed_root / cfg.folder_list_filename, dtype=str))
    if cfg.slide_out in slidename_lst:
        slidename_lst.remove(cfg.slide_out)
    print(cfg.slide_out, "is held out for testing", (cfg.slide_out not in slidename_lst))

    selected_genes = list(np.genfromtxt(processed_root / cfg.gene_list_filename, dtype=str))
    print("Selected gene list loaded. len of selected genes:", len(selected_genes))

    test_adata = anndata.read_h5ad(st_root / f"{cfg.slide_out}.h5ad")
    X = test_adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    test_count_mtx_df = pd.DataFrame(X, columns=test_adata.var_names, index=test_adata.obs_names).loc[
        :, selected_genes
    ]
    test_count_mtx_selected_genes = np.log2(test_count_mtx_df + 1).copy()
    print("Test count mtx shape:", test_count_mtx_selected_genes.shape)
    return selected_genes, test_count_mtx_selected_genes


def load_samples(cfg: EvalConfig):
    pred = torch.load(cfg.sample_path, map_location="cpu")
    if pred.dim() == 3 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    print("Generated samples shape:", tuple(pred.shape))
    return pred


def aggregate_predictions(pred: torch.Tensor, num_spots: int, cfg: EvalConfig, rng: np.random.Generator):
    assert pred.shape[0] >= num_spots * cfg.num_rep, (
        f"Expected at least {num_spots * cfg.num_rep} samples "
        f"({num_spots} spots * {cfg.num_rep} reps), got {pred.shape[0]}"
    )

    random_selected_index = rng.choice(np.arange(cfg.num_rep), cfg.num_selected, replace=False)
    pred_avg = torch.zeros(size=(num_spots, pred.shape[1]), dtype=pred.dtype)
    for i in range(num_spots):
        pred_avg[i] = torch.mean(pred[i * cfg.num_rep + random_selected_index, :], dim=0)
    return pred_avg.cpu().detach().numpy()


def compute_correlations(gt_df: pd.DataFrame, pred_avg: np.ndarray):
    all_corr = []
    for i in range(gt_df.shape[1]):
        x = gt_df.iloc[:, i].values
        y = pred_avg[:, i]

        if y.std() == 0:
            print(f"Warning: std of predicted values for gene {gt_df.columns[i]} is 0!")
        if x.std() == 0:
            print(f"Warning: std of true values for gene {gt_df.columns[i]} is 0! Skipping correlation calculation.")
            continue
        all_corr.append(np.corrcoef(x, y)[0][1])
    return all_corr


def compute_metrics(gt_df: pd.DataFrame, pred_avg: np.ndarray, all_corr: list[float]):
    metrics = {}
    sorted_corr = sorted(all_corr)[::-1]
    metrics["PCC-10"] = float(np.mean(sorted_corr[:10])) if len(sorted_corr) >= 10 else float(np.mean(sorted_corr))
    metrics["PCC-50"] = float(np.mean(sorted_corr[:50])) if len(sorted_corr) >= 50 else float(np.mean(sorted_corr))
    metrics["PCC-max"] = float(np.mean(sorted_corr))  # use all available genes
    metrics["MSE"] = float(np.mean((gt_df.values - pred_avg) ** 2))
    metrics["MAE"] = float(np.mean(np.abs(gt_df.values - pred_avg)))
    pred_var = np.var(pred_avg, axis=0)
    gt_var = np.var(gt_df.values, axis=0)
    gt_var_safe = np.where(gt_var == 0, 1e-8, gt_var)
    metrics["RVD"] = float(np.mean(((pred_var - gt_var_safe) ** 2) / (gt_var_safe**2)))
    return metrics


def save_metrics(metrics: dict, path: Path):
    lines = [f"{k}: {v:.6f}" for k, v in metrics.items()]
    path.write_text("\n".join(lines))
    print(f"Saved metrics to {path}")


def plot_variation_curves(gt_df: pd.DataFrame, pred_avg: np.ndarray, save_path: Path):
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    pred_mean = np.mean(pred_avg, axis=0)
    pred_mean_norm = pred_mean / np.sum(pred_mean)
    gt_mean = np.mean(gt_df, axis=0)
    gt_mean_norm = gt_mean / np.sum(gt_mean)
    gt_mean_sorted = np.sort(gt_mean_norm)
    pred_mean_sorted = pred_mean_norm[np.argsort(gt_mean_norm)]
    axs[0, 0].plot(np.arange(len(gt_mean_sorted)), gt_mean_sorted, label="Ground Truth", c="b")
    axs[0, 0].scatter(np.arange(len(pred_mean_sorted)), pred_mean_sorted, s=5, label="Predicted", c="orange")
    axs[0, 0].set_title("Normalized Mean")
    axs[0, 0].set_xlabel("gene index ordered by mean")
    axs[0, 0].set_ylabel("normalized mean")
    axs[0, 0].legend()

    gt_mean_sorted_abs = np.sort(gt_mean)
    pred_mean_sorted_abs = pred_mean[np.argsort(gt_mean)]
    axs[1, 0].plot(np.arange(len(gt_mean_sorted_abs)), gt_mean_sorted_abs, label="Ground Truth", c="b")
    axs[1, 0].scatter(np.arange(len(pred_mean_sorted_abs)), pred_mean_sorted_abs, s=5, label="Predicted", c="orange")
    axs[1, 0].set_title("Absolute Mean")
    axs[1, 0].set_xlabel("gene index ordered by mean")
    axs[1, 0].set_ylabel("absolute mean")
    axs[1, 0].legend()

    pred_var = np.var(pred_avg, axis=0)
    pred_var_norm = pred_var / np.sum(pred_var)
    gt_var = np.var(gt_df, axis=0)
    gt_var_norm = gt_var / np.sum(gt_var)
    gt_var_sorted = np.sort(gt_var_norm)
    pred_var_sorted = pred_var_norm[np.argsort(gt_var_norm)]
    axs[0, 1].plot(np.arange(len(gt_var_sorted)), gt_var_sorted, label="Ground Truth", c="b")
    axs[0, 1].scatter(np.arange(len(pred_var_sorted)), pred_var_sorted, s=5, label="Predicted", c="orange")
    axs[0, 1].set_title("Normalized Variance")
    axs[0, 1].set_xlabel("gene index ordered by var")
    axs[0, 1].set_ylabel("normalized variance")
    axs[0, 1].legend()

    gt_var_sorted_abs = np.sort(gt_var)
    pred_var_sorted_abs = pred_var[np.argsort(gt_var)]
    axs[1, 1].plot(np.arange(len(gt_var_sorted_abs)), gt_var_sorted_abs, label="Ground Truth", c="b")
    axs[1, 1].scatter(np.arange(len(pred_var_sorted_abs)), pred_var_sorted_abs, s=5, label="Predicted", c="orange")
    axs[1, 1].set_title("Absolute Variance")
    axs[1, 1].set_xlabel("gene index ordered by var")
    axs[1, 1].set_ylabel("absolute variance")
    axs[1, 1].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved variation curves to {save_path}")


def plot_corr_hist(all_corr: list[float], save_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(all_corr, bins=50)
    ax.set_title("Gene-wise PCC distribution")
    ax.set_xlabel("PCC")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved correlation histogram to {save_path}")


def main(cfg: EvalConfig):
    os.makedirs(cfg.results_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    selected_genes, gt_df = load_data(cfg)
    pred = load_samples(cfg)
    pred_avg = aggregate_predictions(pred, gt_df.shape[0], cfg, rng)

    all_corr = compute_correlations(gt_df, pred_avg)
    metrics = compute_metrics(gt_df, pred_avg, all_corr)

    save_metrics(metrics, cfg.results_dir / "metrics.txt")
    plot_corr_hist(all_corr, cfg.results_dir / "corr_hist.png")
    plot_variation_curves(gt_df, pred_avg, cfg.results_dir / "variation_curves.png")


def parse_args() -> Path:
    parser = argparse.ArgumentParser(description="Evaluate generated samples using a TOML config")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        metavar="FILE",
        help="Path to the evaluation TOML config (required)",
    )
    return parser.parse_args().config


def _cli_entrypoint():
    cfg_path: Path = parse_args()
    cfg: EvalConfig = load_toml_config(cfg_path, EvalConfig, ["data", "eval"])
    main(cfg)


if __name__ == "__main__":
    _cli_entrypoint()
