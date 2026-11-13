#!/usr/bin/env bash
# exp05 disk use, by folder; the limit set by the user is 200 GB for all of exp05.
cd "$(dirname "$0")/.."
LIMIT_GB=200
du -sh * 2>/dev/null | sort -h
TOTAL_KB=$(du -s . | cut -f1); TOTAL_GB=$(awk "BEGIN{printf \"%.2f\", $TOTAL_KB/1024/1024}")
echo "exp05 total: ${TOTAL_GB} GB of ${LIMIT_GB} GB"
awk "BEGIN{exit !($TOTAL_KB/1024/1024 > $LIMIT_GB*0.8)}" && echo "WARNING: above 80% of the limit"
exit 0
