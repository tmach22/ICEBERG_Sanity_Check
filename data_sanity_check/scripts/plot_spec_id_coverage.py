"""plot_spec_id_coverage.py

Reads results/spec_id_coverage.json (produced by compute_spec_id_coverage.py)
and renders a bar chart of spec-id coverage per author split file.

Usage:
    .venv/bin/python plot_spec_id_coverage.py
    .venv/bin/python plot_spec_id_coverage.py --in-json <path> --out-png <path>
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
DEFAULT_IN_JSON = SANITY_CHECK_ROOT / "results" / "spec_id_coverage.json"
DEFAULT_OUT_PNG = SANITY_CHECK_ROOT / "results" / "spec_id_coverage.png"

# Single-series bar color (validated categorical slot 1 / blue, light-mode step).
BAR_COLOR = "#2a78d6"
GRID_COLOR = "#d9d8d3"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-json", type=Path, default=DEFAULT_IN_JSON)
    ap.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    args = ap.parse_args()

    with open(args.in_json) as fh:
        summary = json.load(fh)

    splits = summary["splits"]
    # Order: group by dataset_name, then split_file name, for a stable readable axis.
    splits = sorted(splits, key=lambda r: (r["dataset_name"], r["split_file"]))

    labels = [f"{r['dataset_name']}/{r['split_file'].replace('.tsv', '')}" for r in splits]
    coverage = [r["coverage_pct"] for r in splits]
    found = [r["found_in_dataset"] for r in splits]
    total = [r["unique_spec_ids"] for r in splits]

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(labels) + 1.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, coverage, color=BAR_COLOR, height=0.6, zorder=3)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, color=TEXT_PRIMARY, fontsize=10)
    ax.invert_yaxis()  # first split file on top

    ax.set_xlim(0, 100)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.set_xlabel("Spec-id coverage (%)", color=TEXT_SECONDARY, fontsize=10)

    # Recessive gridlines behind bars, no border box.
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(axis="both", length=0, colors=TEXT_SECONDARY)

    # Direct labels: coverage % and (found/total) counts at the end of each bar.
    for bar, pct, f, t in zip(bars, coverage, found, total):
        x = bar.get_width()
        ax.text(
            min(x + 1.5, 96), bar.get_y() + bar.get_height() / 2,
            f"{pct:.2f}%  ({f:,}/{t:,})",
            va="center", ha="left", fontsize=9, color=TEXT_PRIMARY,
        )

    ax.set_title(
        "NIST spec-id coverage: dataset vs. author-provided splits",
        color=TEXT_PRIMARY, fontsize=12, fontweight="bold", loc="left", pad=14,
    )
    fig.text(
        0.01, 0.005,
        f"Dataset: {summary['dataset']['path']}  "
        f"({summary['dataset']['unique_normalized_ids']:,} unique spectrum ids)",
        fontsize=8, color=TEXT_SECONDARY,
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, facecolor=fig.get_facecolor())
    print(f"Wrote plot to {args.out_png}")


if __name__ == "__main__":
    main()
