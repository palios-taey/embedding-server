#!/bin/bash
# Start Qwen3-Embedding-8B Server on DGX Spark
# Optimized for GB10 Blackwell - ~2500 tok/s throughput

CONTAINER_NAME="qwen3-embedding-server"
IMAGE_ID="474ca9e2e7b2"
PORT="${PORT:-8081}"

# Stop existing container if running
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

echo "Starting Qwen3-Embedding Server on port $PORT..."

docker run -d \
  --name $CONTAINER_NAME \
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
  $IMAGE_ID \
  python3 /workspace/server.py

echo "Container started. Waiting for model to load (~70s)..."
echo "Check logs with: docker logs -f $CONTAINER_NAME"
echo "Health check: curl http://localhost:$PORT/health"
echo "Throughput: ~2500 tok/s at batch 512"
