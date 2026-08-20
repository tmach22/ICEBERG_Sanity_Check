"""build_pubchem_formula_map.py

Build the FULL PubChem formula -> {(smiles, inchikey)} map used by
ms-pred's retrieval candidate-pool pipeline, from a downloaded CID-SMILES
file (data/retrieval/pubchem/pubchem_full.txt).

This just calls `build_form_map` from ms-pred's own
data_scripts/pubchem/02_make_formula_subsets.py (imported directly, for
exact parity), but skips that script's tail end which immediately subsets
to one dataset's labels.tsv -- we don't want that dependency here, since
this (the expensive, ~100M-compound part) can run independently of and in
parallel with building our own labels.tsv. Use
data_scripts/pubchem/03_dataset_subset.py separately once labels.tsv exists,
to build the nist20-specific subset from the full map this script produces.

Usage:
    .venv/bin/python build_pubchem_formula_map.py
"""
import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_MS_PRED_ROOT = REPO_ROOT / "vendor" / "ms-pred"
DEFAULT_PUBCHEM_FILE = DEFAULT_MS_PRED_ROOT / "data" / "retrieval" / "pubchem" / "pubchem_full.txt"
DEFAULT_OUT_PICKLE = DEFAULT_MS_PRED_ROOT / "data" / "retrieval" / "pubchem" / "pubchem_formula_map.p"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ms-pred-root", type=Path, default=DEFAULT_MS_PRED_ROOT)
    ap.add_argument("--pubchem-file", type=Path, default=DEFAULT_PUBCHEM_FILE)
    ap.add_argument("--out-pickle", type=Path, default=DEFAULT_OUT_PICKLE)
    args = ap.parse_args()

    sys.path.insert(0, str(args.ms_pred_root / "src"))
    spec = importlib.util.spec_from_file_location(
        "make_formula_subsets", args.ms_pred_root / "data_scripts" / "pubchem" / "02_make_formula_subsets.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec: the multiprocessing pool pickles
    # worker functions by reference (module name + qualname), which requires
    # the module to be discoverable in sys.modules. Without this, dill falls
    # back to pickling the whole module by value, which chokes on RDKit's
    # unpicklable Boost.Python-backed functions imported at module level.
    sys.modules[spec.name] = mod
    # This module's __main__ block only runs under `if __name__ == "__main__"`,
    # which is false when exec'd as a named module -- safe to load without side effects.
    spec.loader.exec_module(mod)

    print(f"Building full PubChem formula map from {args.pubchem_file} ...")
    args.out_pickle.parent.mkdir(parents=True, exist_ok=True)
    mod.build_form_map(smi_file=str(args.pubchem_file), dump_file=str(args.out_pickle), debug=False)
    print(f"Wrote {args.out_pickle}")


if __name__ == "__main__":
    main()
