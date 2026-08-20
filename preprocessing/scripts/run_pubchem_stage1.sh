#!/usr/bin/env bash
# Stage 1 of the retrieval candidate-pool pipeline: download PubChem
# CID-SMILES and build the full formula -> {(smiles, inchikey)} map.
# Runs detached; chains download -> unzip -> formula-map build.
set -euo pipefail

PUBCHEM_DIR=/data/nas-gpu/wang/tmach007/ICEBERG_Sanity_Check/vendor/ms-pred/data/retrieval/pubchem
PY=/data/nas-gpu/wang/tmach007/ICEBERG_Sanity_Check/vendor/ms-pred/.venv/bin/python
SCRIPT_DIR=/data/nas-gpu/wang/tmach007/ICEBERG_Sanity_Check/preprocessing/scripts

echo "[$(date)] Waiting for CID-SMILES.gz download to finish (pid $WGET_PID)..."
while kill -0 "$WGET_PID" 2>/dev/null; do
  sleep 5
done

echo "[$(date)] Download finished. Size: $(du -h "$PUBCHEM_DIR/CID-SMILES.gz" 2>/dev/null | cut -f1)"
echo "[$(date)] Gunzipping..."
gunzip -f "$PUBCHEM_DIR/CID-SMILES.gz"
mv "$PUBCHEM_DIR/CID-SMILES" "$PUBCHEM_DIR/pubchem_full.txt"
echo "[$(date)] pubchem_full.txt ready: $(wc -l < "$PUBCHEM_DIR/pubchem_full.txt") lines"

echo "[$(date)] Building full formula map (this is the long step, many CPU-minutes)..."
"$PY" "$SCRIPT_DIR/build_pubchem_formula_map.py"
echo "[$(date)] DONE"
