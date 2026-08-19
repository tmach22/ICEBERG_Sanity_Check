"""plot_nist20_split_stats.py

Reads results/<split-type>_split_stats/nist20_<split-type>_split_stats.json
(produced by compute_nist20_split_stats.py) and renders a grouped bar chart
comparing the paper's per-fold molecule/spectra counts against our dataset.

Usage:
    .venv/bin/python plot_nist20_split_stats.py --split-type random
    .venv/bin/python plot_nist20_split_stats.py --split-type scaffold
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = Path(__file__).resolve().parent
SANITY_CHECK_ROOT = HERE.parent

FOLD_ORDER = ["train", "val", "test"]
FOLD_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}
METRICS = [("molecules", "Unique molecules"), ("spectra", "Unique spectra")]

SERIES = [
    ("paper", "Paper", "#2a78d6"),
    ("ours", "Ours — Orbitrap HCD only", "#eb6834"),
]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#d9d8d3"
SURFACE = "#fcfcfb"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-type", choices=["random", "scaffold"], required=True)
    ap.add_argument("--in-json", type=Path, default=None)
    ap.add_argument("--out-png", type=Path, default=None)
    args = ap.parse_args()

    results_dir = SANITY_CHECK_ROOT / "results" / f"{args.split_type}_split_stats"
    in_json = args.in_json or results_dir / f"nist20_{args.split_type}_split_stats.json"
    out_png = args.out_png or results_dir / f"nist20_{args.split_type}_split_stats.png"

    with open(in_json) as fh:
        summary = json.load(fh)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    n_series = len(SERIES)
    group_width = 0.62
    bar_width = group_width / n_series
    x = range(len(FOLD_ORDER))

    for ax, (metric_key, metric_title) in zip(axes, METRICS):
        ax.set_facecolor(SURFACE)
        for si, (series_key, series_label, color) in enumerate(SERIES):
            vals = []
            pct_diffs = []
            for fold in FOLD_ORDER:
                comp = summary["by_fold"][fold]["paper_comparison"][metric_key]
                vals.append(comp["paper"] if series_key == "paper" else comp["ours"])
                pct_diffs.append(None if series_key == "paper" else comp["pct_diff"])
            offsets = [xi - group_width / 2 + si * bar_width + bar_width / 2 for xi in x]
            bars = ax.bar(offsets, vals, width=bar_width * 0.9, color=color, zorder=3,
                           label=series_label if ax is axes[0] else None)
            for bar, v, pct in zip(bars, vals, pct_diffs):
                label = f"{v:,}" if pct is None else f"{v:,}\n({pct:+.1f}%)"
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label, ha="center", va="bottom", fontsize=8.5, color=TEXT_PRIMARY,
                    linespacing=1.4,
                )

        ax.set_xticks(list(x))
        ax.set_xticklabels([FOLD_LABELS[f] for f in FOLD_ORDER], color=TEXT_PRIMARY, fontsize=10.5)
        ax.set_title(metric_title, color=TEXT_PRIMARY, fontsize=11, fontweight="bold", pad=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(GRID_COLOR)
        ax.spines["bottom"].set_color(GRID_COLOR)
        ax.tick_params(axis="both", length=0, colors=TEXT_SECONDARY, labelsize=9)
        ax.margins(y=0.24)

    split_label = "Random split" if args.split_type == "random" else "Scaffold split"
    panel = summary.get("paper_panel") or ("a" if args.split_type == "random" else "b")
    fig.suptitle(
        f"NIST'20 {split_label} ({summary['split_file'].split('/')[-1]}): paper vs. our dataset",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", x=0.01, ha="left", y=1.03,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=2, frameon=False, fontsize=9.5, labelcolor=TEXT_PRIMARY,
    )
    fig.text(
        0.01, -0.02,
        "Universe = Orbitrap HCD only, canonical adduct vocabulary, molecule identity = 2D InChIKey. "
        "Fold assigned per molecule from the authors' split file. See methodology in "
        f"nist20_{args.split_type}_split_stats.json.",
        fontsize=7.5, color=TEXT_SECONDARY,
    )

    fig.tight_layout(rect=(0, 0.01, 1, 0.88))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote plot to {out_png}")


if __name__ == "__main__":
    main()
