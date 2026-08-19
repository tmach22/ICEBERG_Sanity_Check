"""plot_precursor_mass_check.py

Reads results/precursor_mass_check/precursor_mass_check.json (produced by
compute_precursor_mass_check.py) and renders a two-bar summary of consistent
vs. outlier spectra (status colors: good / critical), with the outlier bar's
dominant adduct label annotated.

Usage:
    .venv/bin/python plot_precursor_mass_check.py
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
RESULTS_DIR = SANITY_CHECK_ROOT / "results" / "precursor_mass_check"

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#d9d8d3"
SURFACE = "#fcfcfb"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--out-png", type=Path, default=None)
    args = ap.parse_args()
    out_png = args.out_png or args.results_dir / "precursor_mass_check.png"

    with open(args.results_dir / "precursor_mass_check.json") as fh:
        summary = json.load(fh)

    fig, ax_bar = plt.subplots(1, 1, figsize=(6.5, 5.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    ax_bar.set_facecolor(SURFACE)
    n_checked = summary["row_counts"]["rows_checked"]
    n_outliers = summary["outliers"]["count"]
    n_consistent = n_checked - n_outliers
    bars = ax_bar.bar(
        [f"Consistent\n(within {summary['outliers']['threshold_ppm']:.0f} ppm)",
         f"Outliers\n(> {summary['outliers']['threshold_ppm']:.0f} ppm)"],
        [n_consistent, n_outliers],
        color=[STATUS_GOOD, STATUS_CRITICAL], width=0.55, zorder=3,
    )
    for bar, v in zip(bars, [n_consistent, n_outliers]):
        pct = 100.0 * v / n_checked
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{v:,}\n({pct:.2f}%)", ha="center", va="bottom", fontsize=9.5, color=TEXT_PRIMARY,
        )
    ax_bar.set_title("Consistent vs. outlier spectra", color=TEXT_PRIMARY, fontsize=11, fontweight="bold", pad=10)
    ax_bar.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax_bar.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["left"].set_color(GRID_COLOR)
    ax_bar.spines["bottom"].set_color(GRID_COLOR)
    ax_bar.tick_params(axis="both", length=0, colors=TEXT_SECONDARY, labelsize=9)
    ax_bar.margins(y=0.2)

    # Annotate outlier composition if one adduct dominates.
    by_adduct = summary["outliers"].get("by_adduct", {})
    if by_adduct:
        top_adduct, top_n = next(iter(by_adduct.items()))
        top_pct = 100.0 * top_n / max(n_outliers, 1)
        note = f"{top_pct:.1f}% of outliers are adduct '{top_adduct}'"
        ax_bar.text(
            0.5, -0.14, note, transform=ax_bar.transAxes, ha="center", va="top",
            fontsize=8, color=TEXT_SECONDARY,
        )

    fig.suptitle(
        "Precursor m/z consistency check: monoisotopic(SMILES) + adduct vs. recorded precursor_mz",
        color=TEXT_PRIMARY, fontsize=12.5, fontweight="bold", x=0.01, ha="left", y=1.03,
    )
    fig.text(
        0.01, -0.05,
        f"Checked {n_checked:,} of {summary['row_counts']['total_rows']:,} rows (adduct in ms-pred's "
        "canonical ~14-class vocabulary, SMILES parseable). See methodology in precursor_mass_check.json.",
        fontsize=7.5, color=TEXT_SECONDARY,
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote plot to {out_png}")


if __name__ == "__main__":
    main()
