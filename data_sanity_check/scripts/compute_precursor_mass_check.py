"""compute_precursor_mass_check.py

Sanity check: for each spectrum, does the recorded `precursor_mz` agree with
the mass we'd expect from `monoisotopic(SMILES) + adduct_mass_shift`?

    expected_mz = ExactMolWt(smiles) + adduct_mass_shift
    error_ppm   = 1e6 * (precursor_mz - expected_mz) / expected_mz

Adduct mass shifts use the standard proton/electron-mass-corrected adduct
convention (e.g. `[M+H]+` = +1.007276452, not the naive +1.00794 average
atomic mass of H; a neutral-loss adduct like `[M+H-NH3]+` is the protonated
ion minus a neutral NH3, i.e. `+proton - NH3`), based on
`ms_pred.common.chem_utils.ion2mass`.

Scope: only rows whose adduct falls in the canonical ~14-class adduct
vocabulary (singly-charged, single-molecule adducts) are checked -- dimers
("2M+H"), isotope satellites ("M+H+2i"), and multiply-charged species
("M+2H") are excluded, same restriction as compute_nist20_orbitrap_stats.py.
This still covers ~63% of the raw dataset (~643K of ~1.02M rows).

Rows whose |error_ppm| exceeds --outlier-ppm-threshold (default 50 ppm --
generous relative to typical Orbitrap/QTOF accuracy of a few ppm to a few
tens of ppm) are written out individually to outliers.csv for inspection;
they're the rows most likely to have a wrong SMILES, adduct, or precursor
m/z recorded.

Usage:
    .venv/bin/python compute_precursor_mass_check.py
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
SANITY_CHECK_ROOT = HERE.parent
REPO_ROOT = SANITY_CHECK_ROOT.parent

DEFAULT_DATASET_TSV = REPO_ROOT / "NIST20_data" / "NIST20_dataset.tsv"
DEFAULT_MS_PRED_ROOT = REPO_ROOT / "vendor" / "ms-pred"
DEFAULT_OUT_DIR = SANITY_CHECK_ROOT / "results" / "precursor_mass_check"

PPM_THRESHOLDS = [1, 5, 10, 20, 50, 100]

# Adduct aliases that denote a protonated ion which then loses a neutral NH3
# (i.e. shift = +proton - NH3). Recomputed from element masses rather than
# taken as a fixed literal, so it stays consistent with whatever element
# mass table ms-pred is built against.
NH3_LOSS_ADDUCT_ALIASES = ["[M+H-NH3]+", "[M+H-H3N]+", "[M-H3N+H]+", "[M-NH3+H]+"]


def build_adduct_mass_table(ms_pred_root: Path):
    sys.path.insert(0, str(ms_pred_root / "src"))
    from ms_pred.common.chem_utils import ion2mass, ion2onehot_pos, ELEMENT_TO_MASS, ELECTRON_MASS

    adduct_mass = dict(ion2mass)
    nh3_loss_shift = -ELEMENT_TO_MASS["N"] - ELEMENT_TO_MASS["H"] * 2 - ELECTRON_MASS
    for alias in NH3_LOSS_ADDUCT_ALIASES:
        adduct_mass[alias] = nh3_loss_shift

    valid_adducts = set(ion2onehot_pos.keys()) & set(adduct_mass.keys())
    return adduct_mass, valid_adducts


def compute_mass_cache(smiles_values):
    from rdkit.Chem.Descriptors import ExactMolWt
    cache = {}
    for smi in smiles_values:
        if not isinstance(smi, str):
            continue
        mol = Chem.MolFromSmiles(smi)
        cache[smi] = ExactMolWt(mol) if mol is not None else None
    return cache


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-tsv", type=Path, default=DEFAULT_DATASET_TSV)
    ap.add_argument("--ms-pred-root", type=Path, default=DEFAULT_MS_PRED_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--outlier-ppm-threshold", type=float, default=20.0)
    args = ap.parse_args()

    adduct_mass, valid_adducts = build_adduct_mass_table(args.ms_pred_root)

    print(f"Loading {args.dataset_tsv} ...")
    df = pd.read_csv(args.dataset_tsv, sep="\t")
    print(f"  {len(df)} rows")

    print("Computing monoisotopic (exact) mass for all unique SMILES...")
    mass_cache = compute_mass_cache(df["smiles"].unique())
    df["neutral_mass"] = df["smiles"].map(mass_cache)
    n_bad_smiles = df["neutral_mass"].isna().sum()
    print(f"  {n_bad_smiles} rows have an unparseable/missing SMILES.")

    charge_sign = df["ion_mode"].map({"Positive": "+", "Negative": "-"})
    df["canonical_adduct"] = "[" + df["adduct"].astype(str) + "]" + charge_sign.fillna("")
    df["adduct_supported"] = df["canonical_adduct"].isin(valid_adducts)

    n_unsupported = int((~df["adduct_supported"]).sum())
    unsupported_adduct_counts = (
        df.loc[~df["adduct_supported"], "adduct"].value_counts().head(20).to_dict()
    )
    print(f"  {n_unsupported} rows have an adduct outside the canonical 14-class vocabulary "
          f"(dimers, isotope satellites, multiply-charged, etc.) -- excluded from the mass check.")

    checked = df[df["adduct_supported"] & df["neutral_mass"].notna()].copy()
    checked["adduct_mass_shift"] = checked["canonical_adduct"].map(adduct_mass)
    checked["expected_mz"] = checked["neutral_mass"] + checked["adduct_mass_shift"]
    checked["error_da"] = checked["precursor_mz"] - checked["expected_mz"]
    checked["error_ppm"] = 1e6 * checked["error_da"] / checked["expected_mz"]

    n_checked = len(checked)
    print(f"\nChecked {n_checked} rows (adduct supported, SMILES parseable).")

    err = checked["error_ppm"]
    abs_err = err.abs()

    within_thresholds = {
        f"{t}_ppm": {
            "count": int((abs_err <= t).sum()),
            "pct": round(100.0 * (abs_err <= t).mean(), 4),
        }
        for t in PPM_THRESHOLDS
    }

    by_instrument = {}
    for instrument, grp in checked.groupby("instrument"):
        g_abs = grp["error_ppm"].abs()
        by_instrument[instrument] = {
            "n": int(len(grp)),
            "median_abs_ppm": round(float(g_abs.median()), 4),
            "pct_within_5ppm": round(100.0 * (g_abs <= 5).mean(), 4),
            "pct_within_50ppm": round(100.0 * (g_abs <= 50).mean(), 4),
        }

    by_adduct = {}
    for adduct, grp in checked.groupby("adduct"):
        g_abs = grp["error_ppm"].abs()
        by_adduct[adduct] = {
            "n": int(len(grp)),
            "median_abs_ppm": round(float(g_abs.median()), 4),
            "pct_within_5ppm": round(100.0 * (g_abs <= 5).mean(), 4),
        }

    outliers = checked[abs_err > args.outlier_ppm_threshold].sort_values("error_ppm", key=lambda s: s.abs(), ascending=False)
    outlier_cols = [
        "compound_name", "spectrum_id", "smiles", "adduct", "ion_mode", "instrument",
        "collision_energy", "precursor_mz", "expected_mz", "error_da", "error_ppm",
    ]
    outlier_adduct_counts = outliers["adduct"].value_counts().to_dict()

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_path": str(args.dataset_tsv.relative_to(REPO_ROOT)),
        "methodology": {
            "expected_mz_formula": "ExactMolWt(RDKit mol from SMILES) + adduct_mass_shift[canonical_adduct]",
            "error_ppm_formula": "1e6 * (precursor_mz - expected_mz) / expected_mz",
            "adduct_scope": "Restricted to the canonical ~14-class adduct vocabulary -- "
                             "singly-charged, single-molecule adducts only. Dimers, isotope "
                             "satellites, and multiply-charged species are excluded (see "
                             "'unsupported_adducts_top20').",
            "outlier_ppm_threshold": args.outlier_ppm_threshold,
        },
        "row_counts": {
            "total_rows": int(len(df)),
            "rows_with_unparseable_smiles": int(n_bad_smiles),
            "rows_with_unsupported_adduct": n_unsupported,
            "rows_checked": n_checked,
        },
        "unsupported_adducts_top20": unsupported_adduct_counts,
        "error_ppm_stats": {
            "mean": round(float(err.mean()), 6),
            "median": round(float(err.median()), 6),
            "std": round(float(err.std()), 6),
            "min": round(float(err.min()), 6),
            "max": round(float(err.max()), 6),
            "p01": round(float(err.quantile(0.01)), 6),
            "p05": round(float(err.quantile(0.05)), 6),
            "p95": round(float(err.quantile(0.95)), 6),
            "p99": round(float(err.quantile(0.99)), 6),
        },
        "within_ppm_thresholds": within_thresholds,
        "by_instrument": by_instrument,
        "by_adduct": by_adduct,
        "outliers": {
            "threshold_ppm": args.outlier_ppm_threshold,
            "count": int(len(outliers)),
            "pct_of_checked": round(100.0 * len(outliers) / n_checked, 4),
            "by_adduct": outlier_adduct_counts,
            "csv_file": "outliers.csv",
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.out_dir / "precursor_mass_check.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {args.out_dir / 'precursor_mass_check.json'}")

    outliers[outlier_cols].to_csv(args.out_dir / "outliers.csv", index=False)
    print(f"Wrote {args.out_dir / 'outliers.csv'} ({len(outliers)} rows, |ppm| > {args.outlier_ppm_threshold})")

    # Also persist the full per-row error table (for the plotting script / further analysis).
    checked_cols = [
        "spectrum_id", "adduct", "ion_mode", "instrument", "precursor_mz", "expected_mz", "error_ppm",
    ]
    checked[checked_cols].to_csv(args.out_dir / "error_ppm_per_row.csv.gz", index=False, compression="gzip")
    print(f"Wrote {args.out_dir / 'error_ppm_per_row.csv.gz'} ({n_checked} rows)")

    print(f"\nSummary: {within_thresholds['1_ppm']['pct']}% of checked rows within 1 ppm; "
          f"{summary['outliers']['pct_of_checked']}% exceed {args.outlier_ppm_threshold} ppm.")


if __name__ == "__main__":
    main()
