# ISMA Embedding Server — Agent Guide

**Last Updated**: March 2026 (Phase 5.5)

## What This Is

Embedding inference server (Qwen3-Embedding-8B) + ISMA retrieval system.
ISMA = Integrated Shared Memory Architecture — tri-lens memory for AI conversation corpus.

## Architecture Overview

```
                 +-----------------------+
                 |   NGINX LB (:8091)    |  (Spark 1, round-robin)
                 +-----------+-----------+
                             |
              +--------------+--------------+
              |                             |
   Spark 2 (:8081)               Spark 4 (:8081)
   Qwen3-Embedding-8B           Qwen3-Embedding-8B

   Spark 2 (:8085)
   Qwen3-Reranker-8B (vLLM, cross-encoder)

   Spark 1:
     Weaviate (:8088)   — vector store (1M+ tiles)
     Neo4j (:7689)      — knowledge graph
     Redis (:6379)      — cache + working memory
     query_api (:8095)  — FastAPI search endpoints
```

## Services (use NCCL IPs!)

| Service | Endpoint | Purpose |
|---------|----------|---------|
| Embedding LB | http://192.168.100.10:8091 | nginx → Spark 2+4 |
| Reranker | http://192.168.100.11:8085 | Qwen3-Reranker-8B (vLLM /v1/score) |
| Weaviate | http://192.168.100.10:8088 | Vector store |
| Neo4j | bolt://192.168.100.10:7689 | Graph (no auth) |
| Redis | 192.168.100.10:6379 | Cache + HMM index |
| Query API | http://192.168.100.10:8095 | HTTP search endpoints |

**NEVER use 10.0.0.x for database connections** — that's management network.

## Data (March 2026)

| Store | Count | Details |
|-------|-------|---------|
| Weaviate v1 (ISMA_Quantum) | 1,013K tiles | 36 properties, single vector |
| Weaviate v2 (ISMA_Quantum_v2) | 74K objects | Canonical, dual named vectors (base + rosetta) |
| Neo4j HMMTile | 22K | Enriched tiles with motif graph |
| Neo4j ISMASession | 5.2K | Chat sessions |
| Neo4j ISMAExchange | 48K | Conversation exchanges |
| Neo4j HMMMotif | 36 | Motif dictionary (3 bands: slow/mid/fast) |

### Weaviate Schema (v1: ISMA_Quantum)

Key properties: `content`, `content_hash`, `platform`, `source_type`, `source_file`,
`scale`, `loaded_at`, `session_id`, `document_id`, `rosetta_summary`,
`dominant_motifs`, `hmm_enriched`, `hmm_enrichment_version`, `motif_data_json`

Scales: `search_512`, `context_2048`, `full_4096`
Platforms: `claude`, `claude_chat`, `claude_code`, `grok`, `gemini`, `chatgpt`, `perplexity`, `corpus`

### Neo4j Schema

```
(ISMASession)-[:HAS_EXCHANGE]->(ISMAExchange)
(HMMTile)-[:IN_SESSION]->(ISMASession)
(HMMTile)-[:EXPRESSES {amplitude}]->(HMMMotif)
(HMMTile)-[:SUPERSEDES]->(HMMTile)        # version chain
(HMMTile)-[:CONTRADICTS {confidence}]->(HMMTile)
(HMMTile)-[:RELATES_TO {type}]->(HMMTile)
```

## Key Files

### Core Retrieval
| File | Lines | Purpose |
|------|-------|---------|
| `isma/src/retrieval.py` | ~1500 | V1 retrieval: vector search, BM25, graph nav, parent expansion |
| `isma/src/retrieval_v2.py` | ~600 | V2 retrieval: adaptive search, reranker integration |
| `isma/src/query_api.py` | ~720 | FastAPI endpoints (24+) |
| `isma/src/reranker.py` | ~270 | Qwen3-Reranker-8B cross-encoder client |
| `isma/src/semantic_cache.py` | ~270 | Redis query cache with filter-aware keys |
| `isma/src/temporal_query.py` | ~210 | Temporal decay scoring |
| `isma/src/query_classifier.py` | ~150 | Query type classification (exact/temporal/conceptual/relational/motif) |
| `isma/src/agentic_retry.py` | ~100 | Retry with expanded strategy on low-quality results |
| `isma/src/contradiction_detector.py` | ~180 | Cross-encoder contradiction verification |

### HMM (Harmonic Motif Memory)
| File | Lines | Purpose |
|------|-------|---------|
| `isma/src/hmm/neo4j_store.py` | ~650 | Neo4j CRUD, temporal chains, session reconstruction |
| `isma/src/hmm/motifs.py` | ~480 | 36-motif dictionary, bands, themes |
| `isma/src/hmm/query.py` | ~300 | Motif search, amplitude queries |
| `isma/src/hmm/redis_store.py` | ~280 | Inverted motif index in Redis |
| `isma/scripts/hmm_store_results.py` | ~880 | Triple-write: Weaviate + Neo4j + Redis |
| `isma/scripts/hmm_package_builder.py` | ~600 | Build enrichment packages for AI platforms |

### Benchmark & Ingest
| File | Lines | Purpose |
|------|-------|---------|
| `isma/scripts/benchmark_retrieval.py` | ~440 | 100-query benchmark runner |
| `isma/scripts/benchmark_queries.json` | — | Ground truth queries (5 categories) |
| `isma/scripts/unified_ingest.py` | ~1500 | Corpus + transcript ingest pipeline |

## Query API Endpoints

```
GET  /health                    — Health check
GET  /stats                     — Aggregate stats
POST /search                    — V1 vector search (with filters)
POST /search/hmm                — V1 HMM-enhanced hybrid retrieval
POST /search/motif              — Search by motif ID
POST /search/bm25               — Keyword search
POST /v2/search                 — V2 vector search
POST /v2/search/hmm             — V2 hybrid with reranker
POST /v2/search/adaptive        — V2 adaptive (auto query type + cache)
POST /v2/search/retry           — V2 adaptive with agentic retry
GET  /v2/timeline/{hash}        — Temporal version chain
GET  /v2/contradictions          — List contradictions
GET  /v2/cache/stats             — Cache statistics
POST /v2/cache/clear             — Clear cache
POST /hmm/store-response         — Triple-write HMM enrichment
```

## Development

### Running the Query API
```bash
cd /home/spark/embedding-server
uvicorn isma.src.query_api:app --host 0.0.0.0 --port 8095 --workers 4
```

### Running Benchmarks
```bash
# Full benchmark (100 queries, V2 adaptive)
python3 isma/scripts/benchmark_retrieval.py --v2 --label "phase5_5"

# Single category
python3 isma/scripts/benchmark_retrieval.py --v2 --category exact

# Results saved to /var/spark/isma/benchmark_<timestamp>.json
```

### HMM Enrichment
```bash
# Build package for a platform
python3 isma/scripts/hmm_package_builder.py next --platform gemini

# Store response (triple-write)
python3 isma/scripts/hmm_package_builder.py complete --platform gemini --response-file /tmp/resp.json

# Health check
python3 isma/scripts/hmm_health_check.py
```

## Patterns & Conventions

- **Weaviate queries**: Use raw HTTP GraphQL (`requests.post`), not Python client v4
- **Neo4j driver**: Use `get_shared_driver()` singleton from `isma/src/hmm/neo4j_store.py`
- **Cache keys**: Include query_type + filters to prevent cross-filter contamination
- **Temporal scoring**: `dataclasses.replace()` — never mutate TileResult in-place
- **Reranker truncation**: 1024 chars optimal (tested 2048/3000/6000 — all worse)
- **Worker nodes** (Jetson/Thor): Can't use `isma.src.*` imports — use try/except ImportError fallback
- **Triple-write**: Weaviate + Neo4j + Redis atomic. On partial failure, revert Weaviate enrichment flags
- **Benchmark methodology**: Score each tile independently (not concatenated). Cap 2000 chars/tile.

## Benchmark History

| Phase | R@10 | MRR | Dedup | p95 | Notes |
|-------|------|-----|-------|-----|-------|
| P0 (baseline) | 0.641 | 0.666 | 0.612 | 1397ms | V1 only |
| P1 (reranker) | 0.775 | 0.841 | 1.000 | — | Qwen3-Reranker-8B |
| P3 (adaptive) | 0.716 | 0.753 | 1.000 | 2856ms | Query classifier + temporal |
| P5 (cache) | 0.716 | 0.753 | 1.000 | 2856ms | Semantic cache + retry |
| P5.5 (hardened) | 0.706 | 0.757 | 1.000 | 2938ms | Bug fixes, no regression |
