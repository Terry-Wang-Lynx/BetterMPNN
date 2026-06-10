#!/bin/bash
# ========================================================
# Merge per-step screening logs into a single CSV
# Run this after all array tasks complete:
#   bash merge_results.sh output_sampling
# ========================================================

set -euo pipefail

OUTPUT_DIR=${1:-output_sampling}

echo "=== Merging screening results from ${OUTPUT_DIR} ==="

MERGED="${OUTPUT_DIR}/screening_log.csv"

# Write header from the first file
FIRST=$(ls "${OUTPUT_DIR}"/screening_log_step*.csv 2>/dev/null | head -1)
if [ -z "${FIRST}" ]; then
    echo "ERROR: No per-step CSV files found in ${OUTPUT_DIR}"
    exit 1
fi

head -1 "${FIRST}" > "${MERGED}"

# Append data (skip header) from all step files, sorted
for f in "${OUTPUT_DIR}"/screening_log_step*.csv; do
    tail -n +2 "${f}" >> "${MERGED}"
done

# Count results: locate the 'final_pass' column by header name and count rows
# where it equals PASS (the CSV has several columns that can hold "PASS").
read -r TOTAL PASSED < <(awk -F',' '
    NR==1 { for (i=1;i<=NF;i++) if ($i=="final_pass") col=i; next }
    { total++; if (col && $col=="PASS") passed++ }
    END { printf "%d %d\n", total+0, passed+0 }
' "${MERGED}")

echo "Merged ${TOTAL} seed records into ${MERGED}"
echo "  Passed (final_pass): ${PASSED}"
if [ "${TOTAL}" -gt 0 ]; then
    awk -v p="${PASSED}" -v t="${TOTAL}" 'BEGIN { printf "  Pass rate: %.2f%%\n", 100*p/t }'
else
    echo "  Pass rate: n/a (no records)"
fi

# Merge summaries
python3 -c "
import json, glob, os

output_dir = '${OUTPUT_DIR}'
summaries = sorted(glob.glob(os.path.join(output_dir, 'screening_summary_step*.json')))
total_seeds = 0
total_passed = 0

for s in summaries:
    with open(s) as f:
        d = json.load(f)
    total_seeds += d.get('total_seeds_evaluated', 0)
    total_passed += d.get('passed_seeds', 0)

# Load any summary for template
if summaries:
    with open(summaries[0]) as f:
        merged = json.load(f)
    merged['total_seeds_evaluated'] = total_seeds
    merged['passed_seeds'] = total_passed
    merged['pass_rate'] = f'{100*total_passed/max(1,total_seeds):.2f}%'
    merged['num_workers'] = len(summaries)

    out = os.path.join(output_dir, 'screening_summary.json')
    with open(out, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f'Merged summary saved to {out}')
"

echo "=== Merge complete ==="
