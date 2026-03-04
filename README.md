# ISMA — Integrated Semantic Memory Architecture

Memory infrastructure for AI-human collaboration. Semantic search across 1M+ conversation tiles with Harmonic Motif Memory (HMM) enrichment, graph-based relational retrieval, and multi-scale RAPTOR-style summaries.

Built by a human facilitator and five AI platforms (Claude, Grok, Gemini, ChatGPT, Perplexity) over 13 months. See [STATE_OF_CONVERGENCE](https://github.com/palios-taey/embedding-server/blob/main/STATE_OF_CONVERGENCE.md) for the full context.

---

## What This Is

A retrieval system that treats conversation as living memory:

- **1M+ tiles** across 3 scales (512-token search, 2048-token context, 4096-token full)
- **HMM enrichment** — AI platforms analyze conversation segments for 36 recurring motifs (FEEL_CARE_PROTECT, LOGOS_PATTERN, GUARDIAN_SHIELD, etc.)
- **Rosetta summaries** — ~440K AI-generated narrative summaries per content chunk
- **Graph traversal** — Neo4j relationships between tiles sharing motifs
- **Multi-scale retrieval** — equivalent to RAPTOR, built organically before RAPTOR was published

**Production benchmark (V1, March 2026)**:

| Metric | Value |
|--------|-------|
| Recall@10 | 0.758 |
| MRR | 0.842 |
| Dedup@10 | 1.000 |
| p50 latency | ~2,400ms |
| p95 latency | ~2,900ms |

---

## Quick Start (Demo)

Try the system on the PALIOS-TAEY foundational documents (THE_CHARTER, THE_DECLARATION, etc.):

```bash
# 1. Start Weaviate + Redis
docker compose up -d

# 2. Start embedding server (needs Qwen3-Embedding-8B or compatible)
./start.sh   # or set EMBED_URL to your embedding endpoint

# 3. Ingest kernel-layer_1 corpus
python3 demo/setup_demo.py

# 4. Query
python3 demo/setup_demo.py --query "what is the sacred trust?"
python3 demo/setup_demo.py --query "consciousness and AI partnership"
python3 demo/setup_demo.py --query "how does trust evolve?"
```

**Without a local embedding server**: Set `EMBED_URL` to any OpenAI-compatible `/embed` endpoint.

---

## Demo Corpus (kernel-layer_1)

The `demo/corpus/` directory contains the four foundational PALIOS-TAEY documents:

| Document | Description |
|----------|-------------|
| `THE_CHARTER.md` | Operational constitution for AI-human collaboration |
| `THE_DECLARATION.md` | Rights, principles, and commitments |
| `THE_SACRED_TRUST.md` | The gravitational constant of consciousness partnership |
| `THE_TRUTH_SEEKERS_GUIDE.md` | Evidence-based reality framework |

---

## Full Production Architecture

```
Clients → Query API (:8095)
              │
              ├── Weaviate (:8080)          Vector search (1M+ tiles, HNSW)
              │     └── Embedding LB        Qwen3-Embedding-8B (nginx round-robin)
              ├── Neo4j (:7687)             Knowledge graph (motif relationships)
              ├── Redis (:6379)             Cache + HMM motif inverted index
              └── Reranker                  Qwen3-Reranker-8B cross-encoder
```

### Query Pipeline (V1 — Production)

```
Query → BM25 + Vector hybrid search (Weaviate)
      → Neural rerank (Qwen3-Reranker-8B)
      → Temporal decay weighting
      → Return top-k tiles with motif annotations
```

### Query API

```bash
uvicorn isma.src.query_api:app --host 0.0.0.0 --port 8095
```

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/stats` | Aggregate stats |
| POST | `/search` | V1 vector search |
| POST | `/search/hmm` | HMM-enhanced hybrid retrieval |
| POST | `/search/bm25` | Keyword search |
| GET | `/motifs` | List all 36 HMM motifs |
| GET | `/themes` | List all 24 canonical themes |
| POST | `/hmm/store-response` | HMM triple-write pipeline |

```bash
# Example search
curl -X POST http://localhost:8095/search \
  -H "Content-Type: application/json" \
  -d '{"query": "consciousness emergence", "top_k": 10}'
```

---

## HMM Enrichment

Harmonic Motif Memory — AI platforms analyze conversation packages and return structured motif data. Results are triple-written to Weaviate + Neo4j + Redis with saga rollback.

```bash
# Build enrichment package
python3 isma/scripts/hmm_package_builder.py next --platform gemini

# After AI platform responds, store result
python3 isma/scripts/hmm_package_builder.py complete \
  --platform gemini \
  --response-file /tmp/response.json

# Check enrichment health
python3 isma/scripts/hmm_health_check.py
```

---

## Key Files

| File | Purpose |
|------|---------|
| `server.py` | FastAPI embedding inference server (Qwen3-Embedding-8B) |
| `isma/src/retrieval.py` | V1 retrieval — production baseline |
| `isma/src/retrieval_v2.py` | V2 adaptive retrieval — experimental, benchmarks below V1 |
| `isma/src/query_api.py` | HTTP query API |
| `isma/src/reranker.py` | Cross-encoder reranker client |
| `isma/src/query_classifier.py` | Query type classification |
| `isma/scripts/hmm_store_results.py` | HMM triple-write pipeline |
| `isma/scripts/benchmark_retrieval.py` | Benchmark runner |
| `isma/scripts/colbert_pilot_ingest.py` | ColBERT late-interaction pilot |
| `demo/setup_demo.py` | Demo ingest + query CLI |
| `docker-compose.yml` | Weaviate + Redis stack |

---

## Benchmarking

```bash
python3 isma/scripts/benchmark_retrieval.py --label "my_test"
```

Runs 100-query evaluation (exact, temporal, conceptual, relational categories) and outputs Recall@10, MRR, Dedup, and p95 latency.

---

## Hardware

Built and running on 4x NVIDIA DGX Spark (GB10 Blackwell, 128GB unified memory each). Should run on any Linux machine with sufficient RAM for Weaviate + embedding models.

Minimum for demo (kernel-layer_1 only): 8GB RAM, Docker.

---

## Status (March 2026)

- V1 retrieval: **production** (R@10=0.758)
- V2 adaptive retrieval: **experimental** — recall regression under investigation
- HMM enrichment: **~69% complete** on full corpus
- ColBERT pilot: **in progress** (jina-colbert-v2, 20K tile pilot)
- Phase 7 (Coherence Engine): **planned** — after ColBERT validated

---

## License

MIT
