#!/bin/bash
# Start multiple Qwen3-Embedding-8B Servers sequentially
# Waits for each to be healthy before starting next (avoids CUDA memory contention)

IMAGE_ID="nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev"
NUM_INSTANCES=${NUM_INSTANCES:-4}
BASE_PORT=${BASE_PORT:-8081}

echo "Starting $NUM_INSTANCES embedding server instances (sequentially)..."

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
    $IMAGE_ID \
    python3 /workspace/server.py

  # Wait for this instance to become healthy
  echo "  Waiting for instance $i to be healthy..."
  for attempt in $(seq 1 60); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
      echo "  Instance $i is healthy!"
      break
    fi
    sleep 3
  done

  # Verify it's healthy before continuing
  if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "  ERROR: Instance $i failed to start!"
    docker logs $NAME --tail 20
    exit 1
  fi
done

echo ""
echo "All $NUM_INSTANCES instances started on ports $BASE_PORT-$((BASE_PORT + NUM_INSTANCES - 1))"
echo ""
echo "Health checks:"
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  echo "  curl http://localhost:$((BASE_PORT + i))/health"
done
