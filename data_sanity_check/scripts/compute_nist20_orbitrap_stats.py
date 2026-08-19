"""compute_nist20_orbitrap_stats.py

Sanity check: does our dataset reproduce the "Extended Data Table 2" statistics
from the ICEBERG paper ("Statistics of the Orbitrap spectra from the NIST'20
library")?

    Mode    Unique molecules   Unique spectra        Unique spectra
                                (molecules+adducts)   (molecules+adducts+CE)
    ESI+    24,403             35,129                391,288
    ESI-    10,959             14,035                139,352
    Total   25,541             49,164                530,640

Definitions used here, chosen to match the ms-pred authors' own conventions
found in the vendored submodule rather than guessed from scratch:

  * "Unique molecule" identity = 2D InChIKey (first block of the standard
    InChIKey, i.e. connectivity only, stereo/isotope-insensitive). This is
    exactly the identity `data_scripts/make_splits.py` uses for "2D InChIKey
    matching" when building/preserving splits.
  * "Valid adduct" = an adduct that maps into `ms_pred.common.chem_utils.
    ion2onehot_pos`, the fixed ~14-entry canonical adduct vocabulary the
    released models were trained on. Our raw dataset's `adduct` column is far
    more permissive (dimers, isotope satellites, arbitrary neutral losses --
    e.g. "2M+H", "M+H+2i", "M+H-C4H8"); rows with an adduct outside this
    vocabulary are excluded, mirroring what the authors' preprocessing must
    do since their released model only has embedding slots for these 14.
  * "Orbitrap" instrument: NIST'20 records two Orbitrap fragmentation methods
    in the raw `instrument` column -- "HCD" (Orbitrap HCD) and "IT-FT/ion
    trap with FTMS" (Orbitrap CID) -- mapped to `ms_pred`'s "Orbitrap" and
    "IT-FT" instrument classes respectively (see
    `chem_utils.instrument2onehot_pos`). This script restricts to "HCD" only
    (Orbitrap HCD), matching `ion2onehot_pos`'s own inline comment
    (`"Orbitrap": 0,  # Orbitrap HCD`) treating "Orbitrap" as the HCD class
    specifically. "IT-FT" (CID) and "Q-TOF" rows are both excluded.

We could not get an exact match to the published counts (see results) -- the
authors' actual raw-SDF -> labels.tsv conversion happens in a *separate*,
unvendored repo (`rogerwwww/ms-data-parser`, referenced but not included in
`ms-pred`), which likely applies additional filtering (dedup rules, spectral
quality/purity thresholds, etc.) that we can't inspect. This script gets as
close as it can from first principles and reports the deltas honestly rather
than curve-fitting the filters to hit the target numbers.

Usage:
    .venv/bin/python compute_nist20_orbitrap_stats.py
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
DEFAULT_OUT_JSON = SANITY_CHECK_ROOT / "results" / "nist20_orbitrap_stats" / "nist20_orbitrap_stats.json"

# Extended Data Table 2 (ICEBERG paper), Orbitrap spectra from the NIST'20 library.
PAPER_TARGETS = {
    "molecules": {"Positive": 24403, "Negative": 10959, "total": 25541},
    "mol_adduct": {"Positive": 35129, "Negative": 14035, "total": 49164},
    "mol_adduct_ce": {"Positive": 391288, "Negative": 139352, "total": 530640},
}

# Raw NIST'20 `instrument` values that correspond to Orbitrap fragmentation.
# (Only HCD is included -- see methodology note below on why IT-FT/CID was dropped.)
INSTRUMENT_VARIANTS = {
    "orbitrap_hcd_only": ["HCD"],
}


def load_ms_pred_ion2onehot_pos(ms_pred_root: Path):
    sys.path.insert(0, str(ms_pred_root / "src"))
    from ms_pred.common.chem_utils import ion2onehot_pos  # noqa: E402
    return ion2onehot_pos


def compute_inchikey_2d_cache(smiles_values):
    cache = {}
    for smi in smiles_values:
        if not isinstance(smi, str):
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            cache[smi] = None
            continue
        ikey = Chem.MolToInchiKey(mol)
        cache[smi] = ikey.split("-")[0] if ikey else None
    return cache


def mode_breakdown(df: pd.DataFrame):
    """Compute the three Extended-Data-Table-2 metrics, split by ion mode."""
    out = {}
    for mode_label, mode_df in [
        ("Positive", df[df.ion_mode == "Positive"]),
        ("Negative", df[df.ion_mode == "Negative"]),
        ("total", df[df.ion_mode.isin(["Positive", "Negative"])]),
    ]:
        out[mode_label] = {
            "unique_molecules": int(mode_df["inchikey_2d"].nunique()),
            "unique_spectra_mol_adduct": int(
                mode_df[["inchikey_2d", "adduct"]].drop_duplicates().shape[0]
            ),
            "unique_spectra_mol_adduct_ce": int(
                mode_df[["inchikey_2d", "adduct", "collision_energy"]].drop_duplicates().shape[0]
            ),
            "row_count": int(len(mode_df)),
        }
    return out


def with_paper_comparison(breakdown: dict):
    metric_key_map = {
        "unique_molecules": "molecules",
        "unique_spectra_mol_adduct": "mol_adduct",
        "unique_spectra_mol_adduct_ce": "mol_adduct_ce",
    }
    for mode_label, metrics in breakdown.items():
        metrics["paper_comparison"] = {}
        for our_key, paper_key in metric_key_map.items():
            ours = metrics[our_key]
            paper = PAPER_TARGETS[paper_key][mode_label]
            metrics["paper_comparison"][our_key] = {
                "ours": ours,
                "paper": paper,
                "delta": ours - paper,
                "pct_diff": round(100.0 * (ours - paper) / paper, 4) if paper else None,
            }
    return breakdown


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-tsv", type=Path, default=DEFAULT_DATASET_TSV)
    ap.add_argument("--ms-pred-root", type=Path, default=DEFAULT_MS_PRED_ROOT)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = ap.parse_args()

    print(f"Loading {args.dataset_tsv} ...")
    df = pd.read_csv(args.dataset_tsv, sep="\t")
    print(f"  {len(df)} rows")

    print("Computing 2D InChIKeys (skeleton identity, matches ms-pred's own split-preservation logic)...")
    smiles_cache = compute_inchikey_2d_cache(df["smiles"].unique())
    df["inchikey_2d"] = df["smiles"].map(smiles_cache)
    n_bad = df["inchikey_2d"].isna().sum()
    if n_bad:
        print(f"  WARNING: {n_bad} rows have an unparseable/missing SMILES and are dropped from all counts.")
    df = df[df["inchikey_2d"].notna()]

    print(f"Loading canonical adduct vocabulary from {args.ms_pred_root} ...")
    ion2onehot_pos = load_ms_pred_ion2onehot_pos(args.ms_pred_root)
    print(f"  {len(set(ion2onehot_pos.values()))} unique adduct classes ({len(ion2onehot_pos)} string aliases)")

    charge_sign = df["ion_mode"].map({"Positive": "+", "Negative": "-"})
    df["canonical_adduct"] = "[" + df["adduct"].astype(str) + "]" + charge_sign.fillna("")
    df["adduct_valid"] = df["canonical_adduct"].isin(ion2onehot_pos.keys())
    n_invalid_adduct = (~df["adduct_valid"]).sum()
    print(f"  {n_invalid_adduct} rows have an adduct outside the model's canonical vocabulary and are excluded.")

    valid_df = df[df["adduct_valid"]]

    results = {}
    for variant_name, instruments in INSTRUMENT_VARIANTS.items():
        sub = valid_df[valid_df["instrument"].isin(instruments)]
        print(f"\nVariant '{variant_name}' (instrument in {instruments}): {len(sub)} rows")
        breakdown = mode_breakdown(sub)
        breakdown = with_paper_comparison(breakdown)
        for mode_label, metrics in breakdown.items():
            print(
                f"  {mode_label:10s}  molecules={metrics['unique_molecules']:7d}"
                f"  mol+adduct={metrics['unique_spectra_mol_adduct']:7d}"
                f"  mol+adduct+CE={metrics['unique_spectra_mol_adduct_ce']:7d}"
            )
        results[variant_name] = {
            "instruments_included": instruments,
            "by_mode": breakdown,
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "ICEBERG paper, Extended Data Table 2: "
                  "Statistics of the Orbitrap spectra from the NIST'20 library.",
        "dataset_path": str(args.dataset_tsv.relative_to(REPO_ROOT)),
        "total_rows_loaded": int(len(df)),
        "rows_dropped_unparseable_smiles": int(n_bad),
        "rows_dropped_invalid_adduct": int(n_invalid_adduct),
        "methodology": {
            "molecule_identity": "2D InChIKey (first block of Chem.MolToInchiKey, connectivity only), "
                                  "matching data_scripts/make_splits.py's 'existing split preservation "
                                  "uses only 2D InChIKey matching'.",
            "valid_adduct_definition": "adduct string wrapped as '[<adduct>]<charge sign>' must be a key "
                                        "in ms_pred.common.chem_utils.ion2onehot_pos (the ~14-class "
                                        "canonical adduct vocabulary the released models embed).",
            "instrument_variants": "The raw 'instrument' column has HCD (Orbitrap HCD) and 'IT-FT/ion "
                                    "trap with FTMS' (Orbitrap CID) as Orbitrap-family methods, and Q-TOF "
                                    "as a separate instrument. Restricted to instrument=='HCD' only, "
                                    "matching ion2onehot_pos's inline comment ('\"Orbitrap\": 0,  # "
                                    "Orbitrap HCD'). IT-FT (CID) and Q-TOF rows are excluded.",
            "caveat": "Exact reproduction of the published counts was not achieved. The authors' raw-SDF "
                      "-> labels.tsv conversion lives in a separate, unvendored repo "
                      "(rogerwwww/ms-data-parser) whose exact filtering rules (dedup, spectral quality "
                      "thresholds, etc.) we cannot inspect from this repo alone.",
        },
        "paper_targets": PAPER_TARGETS,
        "variants": results,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
