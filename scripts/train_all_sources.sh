#!/usr/bin/env bash
# Train k100s2 sequentially for each of the three shielded source spectra.
# Run from the shielding-ml root:
#   bash scripts/train_all_sources.sh
# Optional: override epoch count
#   EPOCHS=10 bash scripts/train_all_sources.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${REPO_ROOT}/../venv/bin/python"
SCRIPT="${REPO_ROOT}/scripts/train_k100s2_multisrc.py"
EPOCHS="${EPOCHS:-25}"

echo "============================================"
echo " k100s2 multi-source training"
echo " epochs=${EPOCHS}  python=${PYTHON}"
echo "============================================"

for SRC in concrete25 steel27 bpe25; do
    echo ""
    echo "--- Source: ${SRC} ---"
    "${PYTHON}" "${SCRIPT}" --source "${SRC}" --epochs "${EPOCHS}"
done

echo ""
echo "All three training runs complete."
