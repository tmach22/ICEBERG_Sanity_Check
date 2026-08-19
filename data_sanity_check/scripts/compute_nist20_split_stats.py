"""compute_nist20_split_stats.py

Sanity check: does our dataset, partitioned by the authors' train/val/test
split files, reproduce the per-fold molecule/spectra counts reported in the
ICEBERG paper's split figure (panel a = random split, panel b = scaffold
split)?

    Random split (split_1.tsv)          Scaffold split (scaffold_1.tsv)
    train  20,676 molecules 429,751 sp  train  17,849 molecules 419,812 sp
    val     2,309 molecules  47,815 sp  val     2,776 molecules  53,000 sp
    test    2,556 molecules  53,074 sp  test    4,916 molecules  57,828 sp

Both panels' totals (25,541 molecules / 530,640 spectra) match Extended Data
Table 2's "Total" row exactly, so this uses the same universe definition
already validated in compute_nist20_orbitrap_stats.py: instrument == "HCD"
(Orbitrap HCD only), adduct restricted to ms-pred's canonical vocabulary,
ion_mode in {Positive, Negative}, molecule identity = 2D InChIKey.

Fold assignment, however, is done at the MOLECULE level (2D InChIKey), not
by directly matching every raw spectrum row against the split file: the
split file only lists one `spec` id per (molecule, adduct) pair (49,164
entries total, matching Extended Data Table 2's "unique spectra
(molecule+adduct)" count) -- it does not enumerate every individual
collision-energy row. So the procedure is:

  1. Match each split-file `spec` id to its one representative raw dataset
     row (by normalized NIST numeric id) to recover that molecule's SMILES.
  2. Compute each matched molecule's 2D InChIKey and build a
     molecule -> fold lookup. (Verified empirically: zero molecules get
     conflicting fold assignments across their multiple spec/adduct
     entries -- the authors' splitter assigns one fold per molecule, exactly
     as documented in data_scripts/make_splits.py.)
  3. Apply that lookup to every row in our filtered universe via the row's
     own 2D InChIKey (independent of adduct/instrument), so all collision
     energies and adducts of a molecule land in the same fold.
  4. Universe rows whose molecule never appeared in the split file (a small
     minority -- see compute_spec_id_coverage.py) are reported as
     "unassigned" and excluded from the per-fold breakdown.

As with compute_nist20_orbitrap_stats.py, this does not reproduce the
published numbers exactly -- see that script's docstring for why (the raw
SDF -> labels.tsv conversion lives in an unvendored external repo). Deltas
are reported plainly rather than tuned away.

Usage:
    .venv/bin/python compute_nist20_split_stats.py --split-type random
    .venv/bin/python compute_nist20_split_stats.py --split-type scaffold
"""
import argparse
import json
import re
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

DATASET_ID_RE = re.compile(r"^NIST(\d+)$", re.IGNORECASE)
SPLIT_ID_RE = re.compile(r"^nist_(\d+)$", re.IGNORECASE)

SPLIT_CONFIGS = {
    "random": {
        "split_file": "split_1.tsv",
        "paper_panel": "a",
        "targets": {
            "train": {"molecules": 20676, "spectra": 429751},
            "val": {"molecules": 2309, "spectra": 47815},
            "test": {"molecules": 2556, "spectra": 53074},
        },
    },
    "scaffold": {
        "split_file": "scaffold_1.tsv",
        "paper_panel": "b",
        "targets": {
            "train": {"molecules": 17849, "spectra": 419812},
            "val": {"molecules": 2776, "spectra": 53000},
            "test": {"molecules": 4916, "spectra": 57828},
        },
    },
}
FOLD_ORDER = ["train", "val", "test"]


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-type", choices=sorted(SPLIT_CONFIGS), required=True)
    ap.add_argument("--dataset-tsv", type=Path, default=DEFAULT_DATASET_TSV)
    ap.add_argument("--ms-pred-root", type=Path, default=DEFAULT_MS_PRED_ROOT)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    cfg = SPLIT_CONFIGS[args.split_type]
    out_json = args.out_json or (
        SANITY_CHECK_ROOT / "results" / f"{args.split_type}_split_stats" / f"nist20_{args.split_type}_split_stats.json"
    )
    split_path = args.ms_pred_root / "data" / "spec_datasets" / "nist20" / "splits" / cfg["split_file"]

    sys.path.insert(0, str(args.ms_pred_root / "src"))
    from ms_pred.common.chem_utils import ion2onehot_pos  # noqa: E402

    print(f"Loading {args.dataset_tsv} ...")
    df = pd.read_csv(args.dataset_tsv, sep="\t")
    df["num_id"] = df["spectrum_id"].astype(str).str.extract(DATASET_ID_RE)

    print("Computing 2D InChIKeys for all unique SMILES in the dataset...")
    smiles_cache = compute_inchikey_2d_cache(df["smiles"].unique())
    df["inchikey_2d"] = df["smiles"].map(smiles_cache)
    n_bad_smiles = df["inchikey_2d"].isna().sum()

    # --- Build the Orbitrap-HCD / canonical-adduct universe (same definition
    # as compute_nist20_orbitrap_stats.py). ---
    charge_sign = df["ion_mode"].map({"Positive": "+", "Negative": "-"})
    df["canonical_adduct"] = "[" + df["adduct"].astype(str) + "]" + charge_sign.fillna("")
    df["adduct_valid"] = df["canonical_adduct"].isin(ion2onehot_pos.keys())
    universe = df[
        (df["instrument"] == "HCD")
        & (df["adduct_valid"])
        & (df["ion_mode"].isin(["Positive", "Negative"]))
        & (df["inchikey_2d"].notna())
    ].copy()
    print(f"Universe (Orbitrap HCD, canonical adduct, ESI+/ESI-): {len(universe)} rows")

    # --- Build molecule -> fold lookup from the split file. ---
    print(f"Loading split file {split_path} ...")
    split_df = pd.read_csv(split_path, sep="\t")
    split_df["num_id"] = split_df["spec"].astype(str).str.extract(SPLIT_ID_RE)
    fold_col = "Fold_0" if "Fold_0" in split_df.columns else split_df.columns[1]

    merged = split_df.merge(df[["num_id", "inchikey_2d"]], on="num_id", how="left")
    n_unmatched_spec = merged["inchikey_2d"].isna().sum()
    mol_to_fold = dict(zip(merged["inchikey_2d"].dropna(), merged.loc[merged["inchikey_2d"].notna(), fold_col]))

    # Sanity-check: verify each molecule maps to exactly one fold across all its spec entries.
    fold_counts_per_mol = merged.dropna(subset=["inchikey_2d"]).groupby("inchikey_2d")[fold_col].nunique()
    n_conflicting = int((fold_counts_per_mol > 1).sum())

    universe["fold"] = universe["inchikey_2d"].map(mol_to_fold)
    n_unassigned = int(universe["fold"].isna().sum())
    print(f"  {n_unmatched_spec} split spec-ids didn't match a dataset row; "
          f"{n_conflicting} molecules had conflicting fold assignments; "
          f"{n_unassigned} universe rows have no fold (molecule absent from split file).")

    per_fold = {}
    for fold in FOLD_ORDER:
        sub = universe[universe["fold"] == fold]
        n_mol = int(sub["inchikey_2d"].nunique())
        n_spec = int(sub[["inchikey_2d", "adduct", "collision_energy"]].drop_duplicates().shape[0])
        target = cfg["targets"][fold]
        per_fold[fold] = {
            "molecules": n_mol,
            "spectra": n_spec,
            "paper_comparison": {
                "molecules": {
                    "ours": n_mol, "paper": target["molecules"],
                    "delta": n_mol - target["molecules"],
                    "pct_diff": round(100.0 * (n_mol - target["molecules"]) / target["molecules"], 4),
                },
                "spectra": {
                    "ours": n_spec, "paper": target["spectra"],
                    "delta": n_spec - target["spectra"],
                    "pct_diff": round(100.0 * (n_spec - target["spectra"]) / target["spectra"], 4),
                },
            },
        }
        print(f"  {fold:6s} molecules={n_mol:6d} (target {target['molecules']:6d})"
              f"   spectra={n_spec:7d} (target {target['spectra']:7d})")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_type": args.split_type,
        "split_file": str(split_path.relative_to(REPO_ROOT)),
        "paper_source": f"ICEBERG paper, split-statistics figure, panel ({cfg['paper_panel']}).",
        "dataset_path": str(args.dataset_tsv.relative_to(REPO_ROOT)),
        "universe_definition": "instrument=='HCD' (Orbitrap HCD only); adduct in ms_pred.common."
                                "chem_utils.ion2onehot_pos canonical vocabulary; ion_mode in "
                                "{Positive, Negative}; molecule identity = 2D InChIKey. Same "
                                "universe as compute_nist20_orbitrap_stats.py.",
        "fold_assignment_method": "Split file 'spec' ids matched to one representative dataset row "
                                   "each (by normalized NIST numeric id) to recover molecule identity, "
                                   "then a molecule(2D InChIKey) -> fold lookup is built and applied to "
                                   "every universe row via its own 2D InChIKey -- not by matching "
                                   "individual spectrum ids row-for-row.",
        "diagnostics": {
            "rows_with_unparseable_smiles": int(n_bad_smiles),
            "split_spec_ids_unmatched_to_dataset": int(n_unmatched_spec),
            "molecules_with_conflicting_fold_assignment": n_conflicting,
            "universe_rows_with_no_fold_assignment": n_unassigned,
            "universe_total_rows": int(len(universe)),
        },
        "caveat": "As in compute_nist20_orbitrap_stats.py, exact reproduction of the published counts "
                  "was not achieved -- the authors' raw-SDF -> labels.tsv conversion happens in a "
                  "separate, unvendored repo (rogerwwww/ms-data-parser) whose exact filtering we can't "
                  "inspect from here.",
        "by_fold": per_fold,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
