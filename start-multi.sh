#!/bin/bash
# Start multiple Qwen3-Embedding-8B Servers on DGX Spark
# 6 instances × ~15GB = 90GB VRAM (leaving headroom from 128GB)

IMAGE_ID="474ca9e2e7b2"
NUM_INSTANCES=${NUM_INSTANCES:-4}
BASE_PORT=${BASE_PORT:-8081}

echo "Starting $NUM_INSTANCES embedding server instances..."

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  PORT=$((BASE_PORT + i))
  NAME="qwen3-embedding-$i"

  # Stop existing if running
  docker stop $NAME 2>/dev/null
  docker rm $NAME 2>/dev/null

  echo "Starting instance $i on port $PORT..."

  docker run -d \
    --name $NAME \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p $PORT:8080 \
    -v /home/spark/embedding-server:/workspace \
    -v /home/spark/.cache/huggingface:/root/.cache/huggingface \
    -e MODEL_NAME=Qwen/Qwen3-Embedding-8B \
    -e MAX_BATCH_SIZE=512 \
    -e MAX_SEQ_LEN=4096 \
    -e USE_COMPILE=false \
    -e CUDA_VISIBLE_DEVICES=0 \
    $IMAGE_ID \
    python3 /workspace/server.py
done

echo ""
echo "Started $NUM_INSTANCES instances on ports $BASE_PORT-$((BASE_PORT + NUM_INSTANCES - 1))"
echo "Waiting for models to load (~70s each, loading in parallel)..."
echo ""
echo "Health checks:"
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  echo "  curl http://localhost:$((BASE_PORT + i))/health"
done
