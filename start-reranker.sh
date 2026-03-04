#!/bin/bash
# Start Qwen3-Reranker-8B on Spark 2 (co-located with embedding)
# Port 8085, FP8 quantization, score-only task
#
# Prerequisites:
#   - Embedding instance on same node restarted with --gpu-memory-utilization 0.50
#   - Enough free memory (~37GB for reranker at 0.30 utilization)
#
# Usage:
#   ssh spark2 "bash /home/spark/embedding-server/start-reranker.sh"

set -e

MODEL="Qwen/Qwen3-Reranker-8B"
PORT=8085
GPU_UTIL=0.30
MAX_MODEL_LEN=8192

echo "Starting Qwen3-Reranker-8B on port $PORT..."
echo "GPU utilization: $GPU_UTIL"
echo "Max model length: $MAX_MODEL_LEN"

nohup vllm serve "$MODEL" \
    --task score \
    --host 0.0.0.0 \
    --port "$PORT" \
    --quantization fp8 \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    > /tmp/vllm_reranker.log 2>&1 &

echo "Reranker PID: $!"
echo "Log: /tmp/vllm_reranker.log"
echo "Waiting for startup..."

# Wait for health endpoint
for i in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Reranker ready on port $PORT"
        exit 0
    fi
    sleep 2
done

echo "ERROR: Reranker failed to start within 240 seconds"
echo "Check log: /tmp/vllm_reranker.log"
exit 1
