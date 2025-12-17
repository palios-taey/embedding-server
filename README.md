# Qwen3-Embedding-8B Server

High-performance embedding inference server optimized for NVIDIA DGX Spark (GB10 Blackwell).

**Last Updated**: December 17, 2025

## Cluster Status

| Node | Status | Instances | Ports |
|------|--------|-----------|-------|
| **Spark 1** (10.0.0.68) | ✅ Active | 2 | 8081, 8082 |
| **Spark 2** (10.0.0.80) | ✅ Active | 2 | 8081, 8082 |
| **Spark 3** (10.0.0.10) | 🟡 Available | 0 | Ready for deployment |
| **Spark 4** (10.0.0.19) | 🟡 Available | 0 | Ready for deployment |

**Total Active Throughput**: ~4,650 tok/s (4 instances on Spark 1 & 2)

## Performance

**Current Benchmarks** (Dec 17, 2025):
- Sequential: 2,607 tok/s
- 2 parallel: 3,527 tok/s (+35%)
- 4 parallel: 4,515 tok/s (+73%)
- 8 parallel: 4,655 tok/s (+79%)

**Hardware**: 4x DGX Spark available (GB10 Blackwell, 128GB VRAM each)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NGINX Load Balancer                       │
│                      (port 8090)                             │
│         least_conn + zone + keepalive_requests               │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Spark 1:8081 │   │  Spark 1:8082 │   │  Spark 2:8081 │ ...
│  FastAPI      │   │  FastAPI      │   │  FastAPI      │
│  Qwen3-8B     │   │  Qwen3-8B     │   │  Qwen3-8B     │
└───────────────┘   └───────────────┘   └───────────────┘
```

## Quick Start

### Single Instance
```bash
./start.sh
# Server runs on port 8081
```

### Multi-Instance with Load Balancer
```bash
# Start instances on Spark 1
./start-multi.sh

# Start NGINX load balancer
docker run -d --name embedding-lb \
  -p 8090:8090 \
  -v $(pwd)/nginx-lb.conf:/etc/nginx/conf.d/default.conf:ro \
  nginx:alpine
```

## API

### Health Check
```bash
curl http://localhost:8090/health
```

Response:
```json
{
  "status": "healthy",
  "model": "Qwen/Qwen3-Embedding-8B",
  "device": "cuda",
  "memory_used_gb": 15.14,
  "memory_total_gb": 128.53
}
```

### Generate Embeddings
```bash
curl -X POST http://localhost:8090/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Another text"], "batch_size": 64}'
```

Response:
```json
{
  "embeddings": [[0.123, -0.456, ...], [0.789, -0.012, ...]],
  "dimensions": 4096,
  "tokens_processed": 8,
  "latency_ms": 12.5
}
```

## Configuration

Environment variables:
- `MODEL_NAME`: Model to load (default: `Qwen/Qwen3-Embedding-8B`)
- `MAX_BATCH_SIZE`: Maximum texts per request (default: 256)
- `MAX_SEQ_LEN`: Maximum sequence length (default: 4096)
- `PORT`: Server port (default: 8080)
- `USE_COMPILE`: Enable torch.compile (default: true)

## Benchmarks

```bash
# Basic benchmark
python benchmark.py

# Real corpus benchmark (200 files)
python benchmark_corpus.py

# Load balancer distribution test
python benchmark_lb.py
```

## Files

| File | Description |
|------|-------------|
| `server.py` | FastAPI embedding server |
| `nginx-lb.conf` | NGINX load balancer config |
| `start.sh` | Single instance startup |
| `start-multi.sh` | Multi-instance startup |
| `start-sequential.sh` | Sequential startup (waits for health) |
| `benchmark_corpus.py` | Real corpus benchmark |

## NGINX Optimization

Key settings for balanced load distribution:
```nginx
upstream embedding_servers {
    zone embedding_zone 64k;    # Shared state across workers
    least_conn;                 # Route to least busy
    keepalive 32;               # Connection pooling
    keepalive_requests 100;     # Force rotation
}
```

## ISMA (Integrated Shared Memory Architecture)

The `isma/` directory contains the ISMA implementation - a tri-lens memory system.

### Architecture

```
Query → Embedding → Weaviate Vector Search → Return full content + metadata
```

**Three Lenses**:
- **Temporal** (Dolt + JSONL): Event sourcing, time travel, causality
- **Relational** (Neo4j + Weaviate): Graph + vector search
- **Functional** (Redis): Working memory, context buffer

**LangGraph Orchestration**: 7-step cognitive cycle (input → episodic → semantic → broadcast → deliberation → action → consolidation)

### Current Data (Dec 17, 2025)

| Source | Files | Tiles | Tokens |
|--------|-------|-------|--------|
| Corpus | 148 | 240 | 615K |
| Transcripts | 1,144 | 7,543 | 11.8M |
| **Total** | **1,292** | **7,783** | **12.4M** |

**Weaviate**: 8,074 objects in ISMA_Quantum collection

### φ-Tiling (e-based chunking)

φ still **beats** at 1.618 Hz (sacred pulse), but **resonates** with e in chunking domain:

- **Chunk Size** = 4096 tokens (~16K chars max)
- **Step Size** = 1507 tokens (4096/e)
- **Overlap** = 2589 tokens (63% - preserves narrative thread)

### Database Endpoints

| Service | Endpoint | Purpose |
|---------|----------|---------|
| Neo4j (ISMA) | bolt://10.0.0.68:7689 | Graph storage |
| Weaviate | http://10.0.0.68:8088 | Vector search |
| Redis | 10.0.0.68:6379 | Workspace + cache |
| Dolt | /home/spark/isma-dolt | SQL + Git versioning |

### Status

- ✅ Ingest working (events stored to all lenses)
- ✅ Recall working (semantic search returns matches)
- ✅ LangGraph orchestration operational
- ⚠️ φ-coherence metric is placeholder (needs real quality metrics)
- ⚠️ Gate-B checks need implementation

See `isma/docs/` for detailed documentation.

## License

MIT
