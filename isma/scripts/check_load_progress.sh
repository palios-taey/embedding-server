#!/bin/bash
# Check transcript loading progress

echo "=== Transcript Loading Progress ==="
echo ""

# Latest log entries
echo "Latest progress:"
tail -5 /tmp/transcript_load_v3.log 2>/dev/null | grep -E "^\[|^\s+\["

# Weaviate count
echo ""
echo -n "Weaviate objects: "
curl -s 'http://10.0.0.68:8088/v1/graphql' -H 'Content-Type: application/json' \
  -d '{"query": "{ Aggregate { ISMA_Quantum { meta { count } } } }"}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data']['Aggregate']['ISMA_Quantum'][0]['meta']['count'])"

# Process status
echo ""
echo -n "Loader process: "
if pgrep -f "load_corpus_v3" > /dev/null; then
    pid=$(pgrep -f "load_corpus_v3")
    cpu=$(ps -p $pid -o %cpu --no-headers 2>/dev/null || echo "N/A")
    echo "Running (PID: $pid, CPU: $cpu%)"
else
    echo "Not running"
fi

# Estimate if progress available
echo ""
latest=$(tail -1 /tmp/transcript_load_v3.log 2>/dev/null | grep -oP '\[\d+/\d+\]' | head -1)
if [ -n "$latest" ]; then
    done=$(echo $latest | grep -oP '^\[\K\d+')
    total=$(echo $latest | grep -oP '/\K\d+')
    pct=$(python3 -c "print(f'{100*$done/$total:.1f}%')")
    echo "Files: $done/$total ($pct complete)"
fi

echo ""
echo "Embedding service:"
curl -s http://10.0.0.68:8090/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Status: {d[\"status\"]}, Memory: {d[\"memory_used_gb\"]:.1f}GB')"
