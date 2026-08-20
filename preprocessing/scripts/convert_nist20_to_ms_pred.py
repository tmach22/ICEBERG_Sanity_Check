"""convert_nist20_to_ms_pred.py

Convert our library.mgf (+ NIST20_dataset.tsv for ion_mode, since library.mgf
has no explicit polarity tag) into the labels.tsv / spec_files.hdf5 pair
ms-pred expects at data/spec_datasets/nist20/.

This follows the same filtering, grouping, and adduct/instrument conventions
as ms-data-parser's reformat_nist_lcmsms_sdf.py (the script the ms-pred
README points to for NIST'20), but starts from already-parsed MGF+TSV data
instead of a raw NIST .SDF export -- we don't have that raw export, only its
already-parsed downstream form. Several constants (ION_MAP,
INSTRUMENT_MAP_NIST20, VALID_ELS, get_els) are imported directly from that
script for exact parity rather than retyped.

Pipeline:
  1. Load ion_mode per spectrum_id from NIST20_dataset.tsv (validated 1:1
     against library.mgf in an earlier check).
  2. Precompute, per unique SMILES: full InChIKey, formula, neutral
     monoisotopic mass (RDKit).
  3. Stream-parse library.mgf. For each record, apply the NIST'20 filter
     (mirrors reformat_nist_lcmsms_sdf.py's fails_filter):
       - instrument == "HCD" (NIST'20 used HCD only)
       - adduct (canonicalized with ion_mode's charge sign) is in
         ms_pred.common.chem_utils.ion2mass's canonical vocabulary
       - neutral monoisotopic mass <= 1500 Da
       - formula elements subset of VALID_ELS
       - SMILES parseable
  4. Group survivors by (full InChIKey, canonical adduct, instrument),
     merging collision energies: peaks recorded at the same integer-rounded
     CE within a group are merged by summing intensity at matching
     4-decimal-rounded m/z (mirrors merge_charges in build_mgf_str).
  5. Assign each group a `spec` id: prefer a member id that's already used
     in the authors' split_1.tsv/scaffold_1.tsv splits (already validated
     against this dataset), for direct compatibility; else the lowest
     numeric id in the group.
  6. Write labels.tsv and spec_files.hdf5 into
     vendor/ms-pred/data/spec_datasets/nist20/.

Usage:
    .venv/bin/python convert_nist20_to_ms_pred.py
"""
import argparse
import importlib.util
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Descriptors import ExactMolWt

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
PREPROCESSING_ROOT = HERE.parent
REPO_ROOT = PREPROCESSING_ROOT.parent

DEFAULT_MGF = REPO_ROOT / "NIST20_data" / "library.mgf"
DEFAULT_TSV = REPO_ROOT / "NIST20_data" / "NIST20_dataset.tsv"
DEFAULT_MS_PRED_ROOT = REPO_ROOT / "vendor" / "ms-pred"
DEFAULT_MS_DATA_PARSER_ROOT = REPO_ROOT / "vendor" / "ms-data-parser"
DEFAULT_OUT_DIR = REPO_ROOT / "vendor" / "ms-pred" / "data" / "spec_datasets" / "nist20"
DEFAULT_SPLITS_DIR = DEFAULT_OUT_DIR / "splits"

MAX_MASS = 1500.0
MGF_TAGS = {"TITLE", "PEPMASS", "CHARGE", "ADDUCT", "SMILES", "COLLISION_ENERGY",
            "INSTRUMENT", "ION_SOURCE", "COMPOUND_NAME", "SPECTRUMID", "SCANS"}


def load_reference_constants(ms_data_parser_root: Path, ms_pred_root: Path):
    """Import ION_MAP / INSTRUMENT_MAP_NIST20 / VALID_ELS / get_els directly
    from ms-data-parser's reformat_nist_lcmsms_sdf.py, for exact parity."""
    sys.path.insert(0, str(ms_pred_root / "src"))  # it does `import ms_pred.common as common`
    spec = importlib.util.spec_from_file_location(
        "reformat_nist_lcmsms_sdf", ms_data_parser_root / "reformat_nist_lcmsms_sdf.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register before exec, for pickle-by-reference safety
    spec.loader.exec_module(mod)
    return mod


def load_ion_mode_map(tsv_path: Path):
    df = pd.read_csv(tsv_path, sep="\t", usecols=["spectrum_id", "ion_mode"], dtype=str)
    return dict(zip(df["spectrum_id"], df["ion_mode"]))


def load_known_split_spec_numeric_ids(splits_dir: Path):
    """Union of normalized numeric ids from split_1.tsv and scaffold_1.tsv,
    so we can prefer these as our merged groups' `spec` id where possible."""
    ids = set()
    split_id_re = re.compile(r"^nist_(\d+)$", re.IGNORECASE)
    for fname in ["split_1.tsv", "scaffold_1.tsv"]:
        path = splits_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        for spec in df["spec"]:
            m = split_id_re.match(str(spec))
            if m:
                ids.add(m.group(1))
    return ids


def build_smiles_property_cache(smiles_values, valid_els):
    """Precompute (inchikey_full, formula, neutral_mass, formula_els_ok) per unique SMILES."""
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    cache = {}
    el_re = re.compile(r"([A-Z][a-z]*)([0-9]*)")
    for smi in smiles_values:
        if not isinstance(smi, str) or smi == "" or smi.lower() == "nan":
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        formula = CalcMolFormula(mol)
        els = {m[0] for m in el_re.findall(formula)}
        els_ok = els.issubset(valid_els)
        inchikey = Chem.MolToInchiKey(mol)
        mass = ExactMolWt(mol)
        cache[smi] = {
            "inchikey": inchikey,
            "formula": re.findall(r"^([^+\-]*)", formula)[0],  # uncharged formula, mirrors uncharged_formula()
            "mass": mass,
            "els_ok": els_ok,
        }
    return cache


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mgf", type=Path, default=DEFAULT_MGF)
    ap.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    ap.add_argument("--ms-pred-root", type=Path, default=DEFAULT_MS_PRED_ROOT)
    ap.add_argument("--ms-data-parser-root", type=Path, default=DEFAULT_MS_DATA_PARSER_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    args = ap.parse_args()

    ref = load_reference_constants(args.ms_data_parser_root, args.ms_pred_root)
    from ms_pred.common.chem_utils import ion2mass
    from ms_pred.common.misc_utils import HDF5Dataset

    ION_MAP = ref.ION_MAP
    INSTRUMENT_MAP = ref.INSTRUMENT_MAP_NIST20  # {"HCD": "Orbitrap"}
    VALID_ELS = ref.VALID_ELS

    print(f"Loading ion_mode map from {args.tsv} ...")
    ion_mode_by_spec = load_ion_mode_map(args.tsv)

    print(f"Loading known split spec ids from {args.splits_dir} ...")
    known_ids = load_known_split_spec_numeric_ids(args.splits_dir)
    print(f"  {len(known_ids)} known spec ids across split_1.tsv + scaffold_1.tsv")

    print("Collecting unique SMILES from TSV and precomputing properties (InChIKey/formula/mass)...")
    tsv_smiles = pd.read_csv(args.tsv, sep="\t", usecols=["smiles"])["smiles"].dropna().unique()
    smi_cache = build_smiles_property_cache(tsv_smiles, VALID_ELS)
    print(f"  {len(smi_cache)} SMILES with usable properties")

    # groups: (inchikey, canonical_adduct, instrument) -> list of member records
    groups = defaultdict(list)

    n_records = 0
    n_kept = 0
    n_drop_instrument = 0
    n_drop_adduct = 0
    n_drop_mass = 0
    n_drop_formula = 0
    n_drop_smiles = 0

    rec = {}
    n_peak_lines = 0
    peak_lines_buffer = []

    print(f"\nStreaming {args.mgf} ...")
    with open(args.mgf) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "BEGIN IONS":
                rec = {}
                peak_lines_buffer = []
                continue
            if line == "END IONS":
                n_records += 1
                if n_records % 200000 == 0:
                    print(f"  ...{n_records} records, {n_kept} kept so far")

                smi = rec.get("SMILES")
                props = smi_cache.get(smi) if smi else None
                sid = rec.get("SPECTRUMID")
                instrument_raw = rec.get("INSTRUMENT")
                adduct_raw = rec.get("ADDUCT")
                ion_mode = ion_mode_by_spec.get(sid)

                keep = True
                if instrument_raw not in INSTRUMENT_MAP:
                    n_drop_instrument += 1
                    keep = False
                if keep:
                    charge_sign = {"Positive": "+", "Negative": "-"}.get(ion_mode)
                    canonical_adduct = f"[{adduct_raw}]{charge_sign}" if charge_sign else None
                    if canonical_adduct is None or canonical_adduct not in ion2mass:
                        n_drop_adduct += 1
                        keep = False
                if keep and (props is None):
                    n_drop_smiles += 1
                    keep = False
                if keep and props["mass"] > MAX_MASS:
                    n_drop_mass += 1
                    keep = False
                if keep and not props["els_ok"]:
                    n_drop_formula += 1
                    keep = False

                if keep:
                    n_kept += 1
                    try:
                        ce = float(rec.get("COLLISION_ENERGY", "nan"))
                    except ValueError:
                        ce = float("nan")
                    peaks = np.array(peak_lines_buffer, dtype=float) if peak_lines_buffer else np.zeros((0, 2))
                    key = (props["inchikey"], canonical_adduct, INSTRUMENT_MAP[instrument_raw])
                    groups[key].append({
                        "spectrum_id": sid,
                        "ce": ce,
                        "peaks": peaks,
                        "smiles": smi,
                        "formula": props["formula"],
                        "precursor": float(rec.get("PEPMASS", "nan")),
                        "compound_name": rec.get("COMPOUND_NAME", ""),
                    })
                continue

            eq = line.find("=")
            if eq > 0 and line[:eq] in MGF_TAGS:
                rec[line[:eq]] = line[eq + 1:]
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peak_lines_buffer.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass

    print(f"\nParsed {n_records} MGF records.")
    print(f"  Kept: {n_kept}")
    print(f"  Dropped -- instrument != HCD: {n_drop_instrument}")
    print(f"  Dropped -- adduct not canonical: {n_drop_adduct}")
    print(f"  Dropped -- SMILES unusable: {n_drop_smiles}")
    print(f"  Dropped -- mass > {MAX_MASS}: {n_drop_mass}")
    print(f"  Dropped -- formula elements outside valid set: {n_drop_formula}")
    print(f"\nGrouped into {len(groups)} (inchikey, adduct, instrument) spec entries.")

    # --- Build merged entries: choose spec id, merge peaks per (rounded) CE ---
    numeric_id_re = re.compile(r"^NIST(\d+)$", re.IGNORECASE)

    labels_rows = []
    ms_blobs = {}

    for (inchikey, canonical_adduct, instrument), members in groups.items():
        # Choose representative spec id: prefer one already known from the author splits.
        member_numeric_ids = []
        chosen_numeric_id = None
        for m in members:
            match = numeric_id_re.match(m["spectrum_id"])
            if not match:
                continue
            num_id = match.group(1)
            member_numeric_ids.append(num_id)
            if num_id in known_ids and chosen_numeric_id is None:
                chosen_numeric_id = num_id
        if chosen_numeric_id is None and member_numeric_ids:
            chosen_numeric_id = min(member_numeric_ids, key=int)
        if chosen_numeric_id is None:
            continue  # no usable id at all; skip (shouldn't happen)
        spec_id = f"nist_{chosen_numeric_id}"

        rep = members[0]
        smiles = rep["smiles"]
        formula = rep["formula"]
        precursor = rep["precursor"]
        compound_name = rep["compound_name"]
        ionization = ION_MAP.get(canonical_adduct, canonical_adduct)

        # Merge peaks per integer-rounded collision energy.
        ce_to_peaks = defaultdict(dict)  # ce_str -> {rounded_mz: summed_intensity}
        for m in members:
            ce_str = f"{m['ce']:.0f}" if not np.isnan(m["ce"]) else "nan"
            bucket = ce_to_peaks[ce_str]
            for mz, inten in m["peaks"]:
                key_mz = round(mz, 4)
                bucket[key_mz] = bucket.get(key_mz, 0.0) + inten

        collision_energies = sorted(ce_to_peaks.keys(), key=lambda x: (x == "nan", float(x) if x != "nan" else 0))

        labels_rows.append({
            "dataset": "nist2020",
            "spec": spec_id,
            "name": compound_name,
            "formula": formula,
            "ionization": ionization,
            "instrument": instrument,
            "smiles": smiles,
            "inchikey": inchikey,
            "precursor": precursor,
            "collision_energies": collision_energies,
        })

        # Build the .ms blob (mirrors ms-data-parser's dump_fn output format).
        header_lines = [
            f">compound {compound_name}",
            f">formula {formula}",
            f">ionization {ionization}",
            f">parentmass {precursor}",
        ]
        comment_fields = {
            "dataset": "nist2020", "spec": spec_id, "instrument": instrument,
            "inchikey": inchikey, "collision_energies": collision_energies,
        }
        comment_lines = [f"#{k} {v}" for k, v in comment_fields.items()]
        peak_sections = []
        for ce_str in collision_energies:
            rows = sorted(ce_to_peaks[ce_str].items())
            section = [f">collision {ce_str}"] + [f"{mz} {inten}" for mz, inten in rows]
            peak_sections.append("\n".join(section))
        out_str = "\n".join(header_lines) + "\n" + "\n".join(comment_lines) + "\n\n" + "\n\n".join(peak_sections)
        ms_blobs[f"{spec_id}.ms"] = out_str

    print(f"\nBuilt {len(labels_rows)} labels.tsv rows / spec_files.hdf5 entries.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels_df = pd.DataFrame(labels_rows)
    labels_path = args.out_dir / "labels.tsv"
    labels_df.to_csv(labels_path, sep="\t", index=False)
    print(f"Wrote {labels_path}")

    hdf5_path = args.out_dir / "spec_files.hdf5"
    h5 = HDF5Dataset(hdf5_path, "w")
    h5.write_dict(ms_blobs)
    h5.close()
    print(f"Wrote {hdf5_path}")


if __name__ == "__main__":
    main()
