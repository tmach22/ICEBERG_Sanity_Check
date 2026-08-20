"""build_nist20_retrieval_candidates.py

Build the retrieval candidate pools for the nist20 test splits, by calling
ms-pred's data_scripts/pubchem/04_make_retrieval_lists.py directly (its
own module-level main() function, restricted to just the two entries we
need instead of its full compute_entries list).

Produces:
    vendor/ms-pred/data/spec_datasets/nist20/retrieval/cands_df_split_1_50.tsv
    vendor/ms-pred/data/spec_datasets/nist20/retrieval/cands_df_scaffold_1_50.tsv

Usage:
    .venv/bin/python build_nist20_retrieval_candidates.py
"""
import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_MS_PRED_ROOT = REPO_ROOT / "vendor" / "ms-pred"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ms-pred-root", type=Path, default=DEFAULT_MS_PRED_ROOT)
    ap.add_argument("--max-k", type=int, default=50)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    from rdkit import rdBase
    from rdkit import RDLogger
    rdBase.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.*")

    sys.path.insert(0, str(args.ms_pred_root / "src"))
    spec = importlib.util.spec_from_file_location(
        "make_retrieval_lists", args.ms_pred_root / "data_scripts" / "pubchem" / "04_make_retrieval_lists.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # required for multiprocessing (spawn) to pickle worker fns by reference
    spec.loader.exec_module(mod)
    mod.debug = False  # main() reads this as a module global; only set inside the script's own __main__ guard

    dataset = "nist20"
    input_map = str(args.ms_pred_root / "data" / "retrieval" / "pubchem" / f"pubchem_formula_map_{dataset}.p")
    input_dataset_folder = args.ms_pred_root / "data" / "spec_datasets" / dataset

    for split_file in ["split_1.tsv", "scaffold_1.tsv"]:
        print(f"\n=== Building retrieval candidates: {dataset} / {split_file} / test / max_k={args.max_k} ===")
        mod.main(
            max_k=args.max_k,
            workers=args.workers,
            input_map=input_map,
            input_dataset_folder=input_dataset_folder,
            split_file=split_file,
            subset="test",
        )
        print(f"=== Done: {split_file} ===")

    print("\nAll done.")


if __name__ == "__main__":
    main()
