#!/usr/bin/env bash
set -euo pipefail

TEAM_NAME="BreakEven"
ZIP_NAME="${TEAM_NAME}_submission.zip"
STAGE_DIR="staging_submission"

echo "=== Packaging Final Submission for ${TEAM_NAME} ==="

rm -rf "${STAGE_DIR}" "${ZIP_NAME}"
mkdir -p "${STAGE_DIR}/output"
mkdir -p "${STAGE_DIR}/code/business_entity_resolution"

echo "1. Linking output files..."
# Hard link to save disk space and time
ln output_final/matching_results.tsv "${STAGE_DIR}/output/matching_results.tsv"
ln output_final/candidate_pairs.tsv "${STAGE_DIR}/output/candidate_pairs.tsv"

echo "2. Copying code directory..."
cp -r code/business_entity_resolution/src "${STAGE_DIR}/code/business_entity_resolution/"
cp code/business_entity_resolution/README.md "${STAGE_DIR}/code/business_entity_resolution/"
cp code/business_entity_resolution/requirements.txt "${STAGE_DIR}/code/business_entity_resolution/"

# The entry points live at the repo root, not under src/, because they are the
# scripts that actually produced output/. The challenge asks that "anyone
# should be able to regenerate both output files ... using only what is in
# this folder", and without these the package is a library with no runnable
# path: no blocking, no scoring, no calibration. src/pipeline.py alone does
# not cover what was run.
mkdir -p "${STAGE_DIR}/code/business_entity_resolution/scripts"
mkdir -p "${STAGE_DIR}/code/business_entity_resolution/tools"
for f in run_test_inference.py train_model.py rescore_submitted.py; do
  [ -f "$f" ] && cp "$f" "${STAGE_DIR}/code/business_entity_resolution/scripts/"
done
for f in tools/calibrate_and_assemble.py tools/enforce_one_to_one.py \
         tools/apply_deduplication.py tools/inspect_country.py; do
  [ -f "$f" ] && cp "$f" "${STAGE_DIR}/code/business_entity_resolution/tools/"
done
# Remove any __pycache__ from the code directory
find "${STAGE_DIR}/code/business_entity_resolution" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "3. Copying Documentation_template.md..."
cp Documentation_template.md "${STAGE_DIR}/Documentation_template.md"

echo "4. Creating zip archive: ${ZIP_NAME}..."
cd "${STAGE_DIR}"
zip -r "../${ZIP_NAME}" output/ code/ Documentation_template.md
cd ..

rm -rf "${STAGE_DIR}"

echo "=== Package created successfully: ${ZIP_NAME} ==="
ls -lh "${ZIP_NAME}"
