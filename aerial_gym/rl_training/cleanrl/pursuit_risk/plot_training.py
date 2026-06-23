"""Plot pursuit risk-model training metrics.

The script reads one or more run directories containing metrics_history.jsonl
and writes two figures:

* Figure_1_loss.png: train/validation total loss.
* Figure_2_risk_quality.png: validation risk-quality metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="Training run directories.")
    parser.add_argument("--output-dir", default=None, help="Directory for Figure_*.png files.")
    return parser.parse_args()


def read_history(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics_history.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing metrics file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty metrics file: {path}")
    return rows


def label_for(run_dir: Path) -> str:
    return run_dir.name.replace("_", " ")


def values(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs = []
    ys = []
    for row in rows:
        if key in row:
            xs.append(int(row.get("epoch", len(xs))))
            ys.append(float(row[key]))
    return xs, ys


def plot_loss(histories: list[tuple[Path, list[dict]]], output_path: Path):
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)
    for run_dir, rows in histories:
        label = label_for(run_dir)
        for key, style in (("train/loss_total", "-"), ("val/loss_total", "--")):
            xs, ys = values(rows, key)
            if xs:
                ax.plot(xs, ys, style, marker="o", linewidth=1.8, label=f"{label} {key}")
    ax.set_title("Risk model supervised loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_quality(histories: list[tuple[Path, list[dict]]], output_path: Path):
    metric_keys = (
        "val/p_loss_50_brier",
        "val/p_loss_150_brier",
        "val/p_loss_50_auroc",
        "val/p_loss_150_auroc",
        "val/severity_50_mae",
        "val/severity_150_mae",
        "val/ranking_pair_accuracy",
    )
    present = [key for key in metric_keys if any(key in row for _run, rows in histories for row in rows)]
    if not present:
        return
    cols = 2
    rows_count = (len(present) + cols - 1) // cols
    fig, axes = plt.subplots(rows_count, cols, figsize=(9.0, 3.2 * rows_count), dpi=160)
    axes_flat = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]
    for ax, key in zip(axes_flat, present):
        for run_dir, rows in histories:
            xs, ys = values(rows, key)
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.8, label=label_for(run_dir))
        ax.set_title(key.replace("val/", ""))
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[len(present) :]:
        ax.axis("off")
    axes_flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()
    run_dirs = [Path(path) for path in args.run_dirs]
    output_dir = Path(args.output_dir) if args.output_dir is not None else run_dirs[-1]
    output_dir.mkdir(parents=True, exist_ok=True)
    histories = [(run_dir, read_history(run_dir)) for run_dir in run_dirs]
    plot_loss(histories, output_dir / "Figure_1_loss.png")
    plot_quality(histories, output_dir / "Figure_2_risk_quality.png")
    print(f"wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
