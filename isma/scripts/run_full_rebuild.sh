#!/bin/bash
# ISMA Full Rebuild - runs unified_ingest.py for all stages sequentially
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="/var/spark/isma/unified_ingest.log"
cd "$SCRIPT_DIR"

echo "=== ISMA Full Rebuild - $(date) ===" > "$LOG"

STAGES="kernel v0 layer_0 chewy chewy-gallery layer_1 layer_2 github-repos expansion_md mira_md mac_all_md spark_loose transcripts"

for stage in $STAGES; do
    echo "" >> "$LOG"
    echo "========== Starting stage: $stage at $(date) ==========" >> "$LOG"
    python3 -u unified_ingest.py --stage "$stage" >> "$LOG" 2>&1
    RC=$?
    echo "========== Finished stage: $stage (rc=$RC) at $(date) ==========" >> "$LOG"
    if [ "$RC" -ne 0 ]; then
        echo "FATAL: Stage $stage failed with rc=$RC" >> "$LOG"
        exit 1
    fi
done

echo "" >> "$LOG"
echo "=== Full rebuild complete at $(date) ===" >> "$LOG"
