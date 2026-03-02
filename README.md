# Embedding Server + ISMA Retrieval System

High-performance embedding inference (Qwen3-Embedding-8B) and retrieval system optimized for NVIDIA DGX Spark (GB10 Blackwell).

**Last Updated**: March 2026 (Phase 5.5 — Hardening)

## Cluster Status

| Node | Role | Ports | Status |
|------|------|-------|--------|
| **Spark 1** (192.168.100.10) | Weaviate, Neo4j, Redis, Query API | 8088, 7689, 6379, 8095 | Active |
| **Spark 2** (192.168.100.11) | Embedding + Reranker inference | 8081, 8085 | Active |
| **Spark 3** (192.168.100.12) | Reserved (Qwen3.5-122B) | — | Available |
| **Spark 4** (192.168.100.13) | Embedding inference | 8081 | Active |

**NGINX Load Balancer**: 192.168.100.10:8091 (round-robin to Spark 2 + 4)

## Architecture

```
Clients → Query API (:8095)
              │
              ├── Weaviate (:8088)          Vector search (1M+ tiles)
              │     └── Embedding LB (:8091)  Qwen3-Embedding-8B
              ├── Neo4j (:7689)             Knowledge graph (48K exchanges)
              ├── Redis (:6379)             Cache + HMM motif index
              └── Reranker (:8085)          Qwen3-Reranker-8B cross-encoder
```

### Retrieval Pipeline (V2 Adaptive)

```
Query → Classify (exact/temporal/conceptual/relational/motif)
      → Cache check (Redis, filter-aware)
      → Vector search (Weaviate, query-type-specific)
      → Neural rerank (Qwen3-Reranker-8B, 1024 char window)
      → Temporal decay (configurable half-life per query type)
      → Return top-k tiles
```

## Quick Start

### Embedding Server
```bash
./start.sh                     # Single instance on :8081
./start-multi.sh               # Multi-instance with LB
```

### Query API
```bash
uvicorn isma.src.query_api:app --host 0.0.0.0 --port 8095 --workers 4
```

### Search
```bash
# Adaptive search (auto query type + reranker + cache)
curl -X POST http://192.168.100.10:8095/v2/search/adaptive \
  -H "Content-Type: application/json" \
  -d '{"query": "consciousness emergence", "top_k": 10}'

# Filtered by platform
curl -X POST http://192.168.100.10:8095/v2/search/adaptive \
  -H "Content-Type: application/json" \
  -d '{"query": "trust evolution", "platform": "gemini", "top_k": 10}'
```

## Data (March 2026)

| Store | Count | Description |
|-------|-------|-------------|
| Weaviate ISMA_Quantum (v1) | 1,013K tiles | Full corpus, 3 scales, 36 properties |
| Weaviate ISMA_Quantum_v2 | 74K objects | Canonical with dual named vectors |
| Neo4j HMMTile | 22K | Enriched tiles with motif assignments |
| Neo4j ISMAExchange | 48K | Conversation exchanges |
| Neo4j ISMASession | 5.2K | Chat sessions across 7 platforms |
| Neo4j HMMMotif | 36 | Motif dictionary (slow/mid/fast bands) |

### Platforms
`claude`, `claude_chat`, `claude_code`, `grok`, `gemini`, `chatgpt`, `perplexity`, `corpus`

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/stats` | Aggregate stats |
| POST | `/search` | V1 vector search |
| POST | `/search/hmm` | V1 HMM-enhanced hybrid |
| POST | `/search/bm25` | Keyword search |
| POST | `/v2/search/adaptive` | V2 adaptive (recommended) |
| POST | `/v2/search/retry` | V2 with agentic retry |
| GET | `/v2/timeline/{hash}` | Temporal version chain |
| GET | `/v2/contradictions` | Contradiction detection |
| POST | `/hmm/store-response` | HMM triple-write |

See `CLAUDE.md` for full endpoint list and development guide.

## Benchmarks (Phase 5.5)

| Metric | Value |
|--------|-------|
| Recall@10 | 0.706 |
| MRR | 0.757 |
| Precision@10 | 0.624 |
| Dedup@10 | 1.000 |
| p50 latency | 2360ms |
| p95 latency | 2938ms |

```bash
python3 isma/scripts/benchmark_retrieval.py --v2 --label "my_test"
```

## HMM Enrichment

Harmonic Motif Memory — AI platforms analyze conversation packages for recurring motifs.
Results stored via triple-write (Weaviate + Neo4j + Redis).

```bash
python3 isma/scripts/hmm_package_builder.py next --platform gemini
python3 isma/scripts/hmm_package_builder.py complete --platform gemini --response-file /tmp/resp.json
python3 isma/scripts/hmm_health_check.py
```

## Key Files

| File | Purpose |
|------|---------|
| `server.py` | FastAPI embedding server |
| `isma/src/retrieval.py` | V1 retrieval layer |
| `isma/src/retrieval_v2.py` | V2 adaptive retrieval |
| `isma/src/query_api.py` | HTTP endpoints |
| `isma/src/reranker.py` | Cross-encoder client |
| `isma/scripts/hmm_store_results.py` | Triple-write pipeline |
| `isma/scripts/benchmark_retrieval.py` | Benchmark runner |
| `CLAUDE.md` | Full agent/auditor guide |

## License

MIT
