"""plot_nist20_orbitrap_stats.py

Reads results/nist20_orbitrap_stats/nist20_orbitrap_stats.json (produced by
compute_nist20_orbitrap_stats.py) and renders a grouped bar chart comparing
the paper's Extended Data Table 2 numbers against our dataset, for both
"Orbitrap" instrument-inclusion variants.

Usage:
    .venv/bin/python plot_nist20_orbitrap_stats.py
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
RESULTS_DIR = SANITY_CHECK_ROOT / "results" / "nist20_orbitrap_stats"
DEFAULT_IN_JSON = RESULTS_DIR / "nist20_orbitrap_stats.json"
DEFAULT_OUT_PNG = RESULTS_DIR / "nist20_orbitrap_stats.png"

# Fixed categorical order, validated adjacent + all-pairs (first three slots).
SERIES = [
    ("paper", "Paper (Extended Data Table 2)", "#2a78d6"),
    ("orbitrap_hcd_only", "Ours — Orbitrap HCD only", "#eb6834"),
]
MODE_ORDER = ["Positive", "Negative", "total"]
MODE_LABELS = {"Positive": "ESI+", "Negative": "ESI-", "total": "Total"}
METRICS = [
    ("unique_molecules", "Unique molecules"),
    ("unique_spectra_mol_adduct", "Unique spectra\n(molecule + adduct)"),
    ("unique_spectra_mol_adduct_ce", "Unique spectra\n(molecule + adduct + collision energy)"),
]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#d9d8d3"
SURFACE = "#fcfcfb"


def get_value(summary, series_key, metric_key, mode_label):
    if series_key == "paper":
        return summary["paper_targets"][
            {"unique_molecules": "molecules",
             "unique_spectra_mol_adduct": "mol_adduct",
             "unique_spectra_mol_adduct_ce": "mol_adduct_ce"}[metric_key]
        ][mode_label]
    return summary["variants"][series_key]["by_mode"][mode_label][metric_key]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-json", type=Path, default=DEFAULT_IN_JSON)
    ap.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    args = ap.parse_args()

    with open(args.in_json) as fh:
        summary = json.load(fh)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    n_series = len(SERIES)
    group_width = 0.72
    bar_width = group_width / n_series
    x = range(len(MODE_ORDER))

    for ax, (metric_key, metric_title) in zip(axes, METRICS):
        ax.set_facecolor(SURFACE)
        for si, (series_key, series_label, color) in enumerate(SERIES):
            vals = [get_value(summary, series_key, metric_key, m) for m in MODE_ORDER]
            offsets = [xi - group_width / 2 + si * bar_width + bar_width / 2 for xi in x]
            bars = ax.bar(offsets, vals, width=bar_width * 0.92, color=color, zorder=3,
                           label=series_label if ax is axes[0] else None)
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:,}",
                    ha="center", va="bottom", fontsize=7.2, color=TEXT_PRIMARY,
                    rotation=90 if v >= 10000 else 0,
                )

        ax.set_xticks(list(x))
        ax.set_xticklabels([MODE_LABELS[m] for m in MODE_ORDER], color=TEXT_PRIMARY, fontsize=10)
        ax.set_title(metric_title, color=TEXT_PRIMARY, fontsize=10.5, fontweight="bold", pad=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(GRID_COLOR)
        ax.spines["bottom"].set_color(GRID_COLOR)
        ax.tick_params(axis="both", length=0, colors=TEXT_SECONDARY, labelsize=8.5)
        ax.margins(y=0.18)

    fig.suptitle(
        "NIST'20 Orbitrap spectra: paper (Extended Data Table 2) vs. our dataset",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", x=0.01, ha="left", y=1.02,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=3, frameon=False, fontsize=9.5, labelcolor=TEXT_PRIMARY,
    )
    fig.text(
        0.01, -0.02,
        "Molecule identity = 2D InChIKey. Adducts restricted to ms-pred's canonical ~14-class "
        "vocabulary. See methodology notes in nist20_orbitrap_stats.json.",
        fontsize=7.5, color=TEXT_SECONDARY,
    )

    fig.tight_layout(rect=(0, 0.01, 1, 0.90))
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote plot to {args.out_png}")


if __name__ == "__main__":
    main()
