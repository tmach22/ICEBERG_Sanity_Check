"""compute_spec_id_coverage.py

Sanity check: does the NIST spectrum dataset we were handed actually contain
the spectra that the ms-pred authors' train/val/test splits expect?

For every split file under `vendor/ms-pred/data/spec_datasets/*/splits/*.tsv`,
this reads the `spec` column (author IDs, e.g. `nist_1102013`), normalizes it
to a bare numeric NIST ID, and checks how many of those IDs are present in the
`spectrum_id` column of our dataset (e.g. `NIST1102013` -> `1102013`).

Writes a single JSON summary to results/spec_id_coverage.json.

Usage:
    .venv/bin/python compute_spec_id_coverage.py
    .venv/bin/python compute_spec_id_coverage.py --dataset-tsv <path> --ms-pred-root <path>
"""
import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SANITY_CHECK_ROOT = HERE.parent
REPO_ROOT = SANITY_CHECK_ROOT.parent

DEFAULT_DATASET_TSV = REPO_ROOT / "NIST20_data" / "NIST20_dataset.tsv"
DEFAULT_MS_PRED_ROOT = REPO_ROOT / "vendor" / "ms-pred"
DEFAULT_OUT_JSON = SANITY_CHECK_ROOT / "results" / "spec_id_coverage.json"

# Dataset ids look like "NIST1102013"; split ids look like "nist_1102013".
# Normalize both down to the bare numeric NIST id so they can be compared.
DATASET_ID_RE = re.compile(r"^NIST(\d+)$", re.IGNORECASE)
SPLIT_ID_RE = re.compile(r"^nist_(\d+)$", re.IGNORECASE)


def normalize_dataset_id(raw: str):
    m = DATASET_ID_RE.match(raw.strip())
    return m.group(1) if m else None


def normalize_split_id(raw: str):
    m = SPLIT_ID_RE.match(raw.strip())
    return m.group(1) if m else None


def load_dataset_ids(dataset_tsv: Path, id_column: str = "spectrum_id"):
    ids = set()
    total_rows = 0
    unparseable = 0
    with open(dataset_tsv, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if id_column not in reader.fieldnames:
            raise SystemExit(
                f"Column '{id_column}' not found in {dataset_tsv}. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            total_rows += 1
            norm = normalize_dataset_id(row[id_column])
            if norm is None:
                unparseable += 1
                continue
            ids.add(norm)
    return ids, total_rows, unparseable


def find_split_files(ms_pred_root: Path, dataset_names=("nist20",), split_file_stems=("split_1", "scaffold_1")):
    splits_root = ms_pred_root / "data" / "spec_datasets"
    paths = sorted(splits_root.glob("*/splits/*.tsv"))
    return [
        p for p in paths
        if p.parent.parent.name in dataset_names and p.stem in split_file_stems
    ]


def check_split_file(split_path: Path, dataset_ids: set):
    total_rows = 0
    unparseable = 0
    seen_ids = set()
    by_fold = {}

    with open(split_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if "spec" not in reader.fieldnames:
            raise SystemExit(f"Column 'spec' not found in {split_path}. Columns: {reader.fieldnames}")
        fold_col = "Fold_0" if "Fold_0" in reader.fieldnames else None

        for row in reader:
            total_rows += 1
            norm = normalize_split_id(row["spec"])
            if norm is None:
                unparseable += 1
                continue
            seen_ids.add(norm)
            fold = row[fold_col] if fold_col else "unknown"
            bucket = by_fold.setdefault(fold, {"total": 0, "found": 0})
            bucket["total"] += 1
            if norm in dataset_ids:
                bucket["found"] += 1

    unique_ids = len(seen_ids)
    found_ids = len(seen_ids & dataset_ids)
    missing_ids = unique_ids - found_ids

    for fold, bucket in by_fold.items():
        bucket["coverage_pct"] = round(100.0 * bucket["found"] / bucket["total"], 4) if bucket["total"] else None

    return {
        "dataset_name": split_path.parent.parent.name,  # e.g. "nist20" / "nist23"
        "split_file": split_path.name,
        "path": str(split_path.relative_to(REPO_ROOT)),
        "total_rows": total_rows,
        "unparseable_rows": unparseable,
        "unique_spec_ids": unique_ids,
        "found_in_dataset": found_ids,
        "missing_from_dataset": missing_ids,
        "coverage_pct": round(100.0 * found_ids / unique_ids, 4) if unique_ids else None,
        "by_fold": by_fold,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-tsv", type=Path, default=DEFAULT_DATASET_TSV)
    ap.add_argument("--id-column", default="spectrum_id")
    ap.add_argument("--ms-pred-root", type=Path, default=DEFAULT_MS_PRED_ROOT)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument(
        "--dataset-names", nargs="+", default=["nist20"],
        help="Which data/spec_datasets/<name>/splits/ subfolders to check (default: nist20 only, "
             "since this dataset is NIST20-derived).",
    )
    ap.add_argument(
        "--split-file-stems", nargs="+", default=["split_1", "scaffold_1"],
        help="Which split file basenames (without .tsv) to check "
             "(default: split_1 and scaffold_1; excludes fingerprint_1/hyperopt).",
    )
    args = ap.parse_args()

    print(f"Loading dataset spectrum ids from {args.dataset_tsv} (column={args.id_column!r})...")
    dataset_ids, dataset_total_rows, dataset_unparseable = load_dataset_ids(args.dataset_tsv, args.id_column)
    print(f"  {dataset_total_rows} rows, {len(dataset_ids)} unique normalized ids, "
          f"{dataset_unparseable} rows failed to parse as a NIST id.")

    split_files = find_split_files(
        args.ms_pred_root,
        dataset_names=tuple(args.dataset_names),
        split_file_stems=tuple(args.split_file_stems),
    )
    if not split_files:
        raise SystemExit(
            f"No split files found under {args.ms_pred_root}/data/spec_datasets/"
            f"{{{','.join(args.dataset_names)}}}/splits/{{{','.join(args.split_file_stems)}}}.tsv"
        )

    print(f"Found {len(split_files)} split file(s) to check.")
    results = []
    for split_path in split_files:
        print(f"  Checking {split_path.relative_to(REPO_ROOT)} ...")
        res = check_split_file(split_path, dataset_ids)
        print(f"    {res['found_in_dataset']}/{res['unique_spec_ids']} ids found "
              f"({res['coverage_pct']}%)")
        results.append(res)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "path": str(args.dataset_tsv.relative_to(REPO_ROOT)) if args.dataset_tsv.is_relative_to(REPO_ROOT) else str(args.dataset_tsv),
            "id_column": args.id_column,
            "total_rows": dataset_total_rows,
            "unique_normalized_ids": len(dataset_ids),
            "unparseable_rows": dataset_unparseable,
        },
        "split_dataset_names_included": list(args.dataset_names),
        "split_file_stems_included": list(args.split_file_stems),
        "normalization": {
            "dataset_id_pattern": DATASET_ID_RE.pattern,
            "split_id_pattern": SPLIT_ID_RE.pattern,
            "note": "Both sides are reduced to the bare numeric NIST id before comparison, "
                    "e.g. dataset 'NIST1102013' and split 'nist_1102013' both normalize to '1102013'.",
        },
        "splits": results,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote coverage summary to {args.out_json}")


if __name__ == "__main__":
    main()
