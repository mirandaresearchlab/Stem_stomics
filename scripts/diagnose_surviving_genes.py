#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

import anndata as ad
import pandas as pd


ENSEMBL_RE = re.compile(r"^ENSG[0-9]+", re.IGNORECASE)


def _find_latest_processed(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = sorted(root.glob("processed_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Quick diagnostics for surviving genes from processed outputs.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Path to processed_<YYYYMMDD> directory. If omitted, picks latest under /storage/hest1k or /proj/.../storage/hest1k.",
    )
    parser.add_argument(
        "--hgnc-path",
        type=Path,
        default=Path("assets/hgnc_complete_set.txt"),
        help="Path to HGNC complete set TSV (hgnc_complete_set.txt). Defaults to assets/hgnc_complete_set.txt.",
    )
    parser.add_argument(
        "--per-file-dir",
        type=Path,
        default=None,
        help="Directory to write per-h5ad drop reports. Defaults to <processed_dir>/gene_filter_reports.",
    )
    args = parser.parse_args()

    processed_dir = args.processed_dir
    if processed_dir is None:
        processed_dir = _find_latest_processed(Path("/storage/hest1k"))
        if processed_dir is None:
            processed_dir = _find_latest_processed(Path("/proj/berzelius-2025-380/storage/hest1k"))
    if processed_dir is None:
        raise SystemExit("Could not locate processed_* directory. Provide --processed-dir.")

    h5ad_dir = processed_dir / "h5ad"
    if not h5ad_dir.exists():
        raise SystemExit(f"Missing h5ad directory: {h5ad_dir}")

    if args.hgnc_path is None or not args.hgnc_path.exists():
        raise SystemExit("Provide --hgnc-path to HGNC complete set TSV (hgnc_complete_set.txt).")

    hgnc = pd.read_csv(args.hgnc_path, sep="\t", dtype=str)
    hgnc_cols = ["symbol", "alias_symbol", "prev_symbol"]
    missing_cols = [c for c in hgnc_cols if c not in hgnc.columns]
    if missing_cols:
        raise SystemExit(f"HGNC file missing columns: {missing_cols}")

    def _explode_col(col: str) -> set[str]:
        series = hgnc[col].dropna().astype(str).str.lower()
        values = set()
        for item in series:
            for token in item.split("|"):
                token = token.strip()
                if token:
                    values.add(token)
        return values

    known_genes = _explode_col("symbol") | _explode_col("alias_symbol") | _explode_col("prev_symbol")

    all_genes: set[str] = set()
    per_file_stats = []
    dropped_prefix_counts = {}
    dropped_prefix_examples = {}
    per_file_dir = args.per_file_dir or (processed_dir / "gene_filter_reports")
    per_file_dir.mkdir(parents=True, exist_ok=True)
    h5ad_paths = sorted(h5ad_dir.glob("*.h5ad"))
    total = len(h5ad_paths)
    if total == 0:
        raise SystemExit(f"No .h5ad files found in {h5ad_dir}")
    for idx, path in enumerate(h5ad_paths, start=1):
        adata = ad.read_h5ad(path, backed="r")
        genes = pd.Index(map(str, adata.var_names)).str.lower()
        all_genes.update(genes)
        is_known = genes.isin(known_genes)
        is_mito = genes.str.startswith("mt-")
        is_ribo = genes.str.startswith(("rpl", "rps"))
        is_ercc = genes.str.startswith("ercc")
        keep_mask = is_known | is_mito | is_ribo | is_ercc
        kept = int(keep_mask.sum())
        total_genes = int(genes.size)
        dropped = total_genes - kept
        drop_pct = 100.0 * dropped / total_genes if total_genes else 0.0
        per_file_stats.append(
            {
                "file": path.name,
                "total_genes": total_genes,
                "kept": kept,
                "dropped": dropped,
                "drop_pct": drop_pct,
            }
        )
        dropped_genes = genes[~keep_mask].tolist()
        for gene in dropped_genes:
            if gene.startswith("ac"):
                prefix = "ac"
            elif gene.startswith("al"):
                prefix = "al"
            elif gene.startswith("ap"):
                prefix = "ap"
            elif gene.startswith("ctd-"):
                prefix = "ctd"
            elif gene.startswith("rp11-"):
                prefix = "rp11"
            elif gene.startswith("fp"):
                prefix = "fp"
            elif gene.startswith("linc"):
                prefix = "linc"
            elif gene.startswith("mir"):
                prefix = "mir"
            elif gene.startswith("snord"):
                prefix = "snord"
            elif gene.startswith("snhg"):
                prefix = "snhg"
            else:
                prefix = "other"
            dropped_prefix_counts[prefix] = dropped_prefix_counts.get(prefix, 0) + 1
            if prefix not in dropped_prefix_examples:
                dropped_prefix_examples[prefix] = gene
        report_path = per_file_dir / f"{path.stem}_gene_filter.txt"
        with report_path.open("w", encoding="utf-8") as f:
            f.write(f"file\t{path.name}\n")
            f.write(f"total_genes\t{total_genes}\n")
            f.write(f"kept\t{kept}\n")
            f.write(f"dropped\t{dropped}\n")
            f.write(f"drop_pct\t{drop_pct:.2f}\n")
            f.write("dropped_genes\n")
            for gene in dropped_genes:
                f.write(f"{gene}\n")
        adata.file.close()
        if idx % 50 == 0 or idx == total:
            print(f"Processed {idx}/{total} h5ad files")

    gene_series = pd.Series(sorted(all_genes))
    gene_patterns = {
        "ensembl": gene_series.str.match(ENSEMBL_RE, na=False).sum(),
        "ercc": gene_series.str.startswith("ercc", na=False).sum(),
        "mito": gene_series.str.startswith("mt-", na=False).sum(),
        "ribosomal": gene_series.str.startswith(("rpl", "rps"), na=False).sum(),
        "numeric_only": gene_series.str.match(r"^[0-9]+$", na=False).sum(),
    }

    print(f"Processed dir: {processed_dir}")
    print(f"Total unique genes: {len(all_genes)}")
    print("Pattern counts:")
    for key, val in gene_patterns.items():
        print(f"  {key}: {val}")

    stats_df = pd.DataFrame(per_file_stats).sort_values("drop_pct", ascending=False)
    print("\nPer-file drop rates (top 20 by drop_pct):")
    print(stats_df.head(20).to_string(index=False))

    high_drop = stats_df[stats_df["drop_pct"] > 50.0]
    if not high_drop.empty:
        print("\nFiles with >50% drop:")
        print(high_drop.to_string(index=False))
    else:
        print("\nNo files with >50% drop.")

    if dropped_prefix_counts:
        prefix_series = pd.Series(dropped_prefix_counts).sort_values(ascending=False)
        print("\nMost common dropped prefix classes:")
        print(prefix_series.to_string())
        print("\nPrefix examples:")
        for prefix, example in dropped_prefix_examples.items():
            print(f"  {prefix}: {example}")


if __name__ == "__main__":
    main()
