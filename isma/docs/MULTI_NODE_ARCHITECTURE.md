# ISMA Multi-Node Architecture
**Target: 4 Sparks + 2 Thors**

## Current State (Pre-Switch)

| Node | IP | Status | Services |
|------|-----|--------|----------|
| **Spark 1** | 10.0.0.68 | Active | Neo4j, Redis, Embedding×2, Load Balancer |
| **Spark 2** | 10.0.0.80 | Active | Weaviate, Embedding×2 |
| **Spark 3** | TBD | Pending switch | - |
| **Spark 4** | TBD | Pending switch | - |
| **Thor 1** | 10.0.0.93 | Standby | Dev kit |
| **Thor 2** | 10.0.0.78 | Standby | Dev kit |

---

## Target Architecture (Post-Switch)

### Service Distribution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ISMA DISTRIBUTED ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SPARK 1 (10.0.0.68)           SPARK 2 (10.0.0.80)                          │
│  ═══════════════════           ═══════════════════                          │
│  ROLE: ISMA Core               ROLE: Vector Store                            │
│  ├─ Neo4j (7687)               ├─ Weaviate (8080)                           │
│  ├─ Redis (6379)               ├─ Embedding×2 (8081,8082)                   │
│  ├─ Dolt (3306)                └─ Weaviate backup                           │
│  ├─ ISMA Daemons                                                            │
│  │   ├─ redis_bridge.py                                                     │
│  │   ├─ breathing_cycle.py                                                  │
│  │   └─ consolidation_worker.py                                             │
│  └─ MCP Server (8100)                                                       │
│                                                                              │
│  SPARK 3 (TBD)                 SPARK 4 (TBD)                                │
│  ═══════════════════           ═══════════════════                          │
│  ROLE: Entity Extraction       ROLE: Orchestration                          │
│  ├─ Zep (8000)                 ├─ LangGraph Server                          │
│  ├─ Embedding×2 (8081,8082)    ├─ Cognitive Cycle Daemon                    │
│  └─ PostgreSQL (for Zep)       └─ System 2 Reasoning                        │
│                                                                              │
│  THOR 1 (10.0.0.93)            THOR 2 (10.0.0.78)                           │
│  ═══════════════════           ═══════════════════                          │
│  ROLE: Edge/Development        ROLE: Edge/Testing                           │
│  ├─ Test deployments           ├─ Integration tests                         │
│  ├─ Dev experiments            ├─ Load testing                              │
│  └─ Staging                    └─ Chaos engineering                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Network Topology

```
                    ┌─────────────────────────┐
                    │      10GbE Switch       │
                    │   (New - arriving today)│
                    └───────────┬─────────────┘
                                │
        ┌───────────┬───────────┼───────────┬───────────┐
        │           │           │           │           │
   ┌────▼────┐ ┌────▼────┐ ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
   │ Spark 1 │ │ Spark 2 │ │ Spark 3 │ │ Spark 4 │ │  Thors  │
   │  Core   │ │ Vectors │ │ Entities│ │  Orch   │ │  Edge   │
   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
        │           │           │           │
        └───────────┴─────┬─────┴───────────┘
                          │
                 200GbE Inter-Spark
                 (existing point-to-point)
```

---

## Service Specifications

### Spark 1: ISMA Core (The Brain)

| Service | Port | Purpose | Resources |
|---------|------|---------|-----------|
| **Neo4j** | 7687 | Knowledge graph, entities, relationships | 16GB heap |
| **Redis** | 6379 | Cache, pub/sub, functional lens | 8GB |
| **Dolt** | 3306 | Versioned temporal events | 4GB |
| **ISMA Daemons** | - | Background processing | 4GB |
| **MCP Server** | 8100 | Claude Code integration | 2GB |

**Total Estimated**: ~34GB RAM, minimal GPU

### Spark 2: Vector Store

| Service | Port | Purpose | Resources |
|---------|------|---------|-----------|
| **Weaviate** | 8080 | Semantic vector storage | 16GB |
| **Embedding×2** | 8081-8082 | Qwen3-Embedding-8B | ~30GB GPU each |

**Total Estimated**: ~16GB RAM, ~60GB GPU

### Spark 3: Entity Extraction

| Service | Port | Purpose | Resources |
|---------|------|---------|-----------|
| **Zep** | 8000 | Automatic entity extraction | 8GB |
| **PostgreSQL** | 5432 | Zep backend | 4GB |
| **Embedding×2** | 8081-8082 | Additional capacity | ~30GB GPU each |

**Total Estimated**: ~12GB RAM, ~60GB GPU

### Spark 4: Orchestration

| Service | Port | Purpose | Resources |
|---------|------|---------|-----------|
| **LangGraph** | 8200 | Cognitive cycle orchestration | 8GB |
| **System 2** | 8201 | Deliberation, branching | 8GB |
| **Checkpointer** | - | LangGraph state persistence | 4GB |

**Total Estimated**: ~20GB RAM, GPU for LLM inference if needed

---

## Migration Plan

### Phase 1: Switch Installation (Today)
1. Physical switch connection
2. IP assignment for Spark 3, Spark 4
3. Network verification (ping, bandwidth)
4. Update `/etc/hosts` on all nodes

### Phase 2: Service Migration
```
Current → Target

Spark 1:
  Keep: Neo4j, Redis
  Add: Dolt, ISMA daemons, MCP server
  Move: Embedding instances → Spark 3

Spark 2:
  Keep: Weaviate, Embedding×2
  Add: Weaviate replication config

Spark 3 (new):
  Add: Embedding×2 (from Spark 1)
  Add: Zep, PostgreSQL

Spark 4 (new):
  Add: LangGraph orchestration
  Add: System 2 reasoning
```

### Phase 3: Integration Verification
1. Test cross-node connectivity
2. Update nginx load balancer for 6 embedding instances
3. Verify ISMA can reach all services
4. Run full cognitive cycle test

---

## Load Balancer Update

```nginx
# /home/spark/embedding-server/nginx-lb.conf (updated for 6 instances)

upstream embedding_servers {
    least_conn;

    # Spark 1 (moved to Spark 3 after migration)
    # server 10.0.0.68:8081 weight=1;
    # server 10.0.0.68:8082 weight=1;

    # Spark 2 (vectors node)
    server 10.0.0.80:8081 weight=1;
    server 10.0.0.80:8082 weight=1;

    # Spark 3 (entity extraction node) - ADD AFTER SWITCH
    # server SPARK3_IP:8081 weight=1;
    # server SPARK3_IP:8082 weight=1;
}
```

---

## Scaling Benefits

| Metric | Current (2 Sparks) | Target (4 Sparks + 2 Thors) |
|--------|-------------------|----------------------------|
| **Embedding throughput** | 4 instances (~5,000 tok/s) | 6 instances (~7,500 tok/s) |
| **ISMA resilience** | Single point of failure | Distributed, redundant |
| **Entity extraction** | Manual | Automatic (Zep) |
| **Orchestration** | None | LangGraph cognitive cycle |
| **Development** | Shares prod | Dedicated Thor nodes |

---

## IP Assignment Plan (Pending)

| Node | Proposed IP | Notes |
|------|-------------|-------|
| Spark 3 | 10.0.0.81 | Sequential after Spark 2 |
| Spark 4 | 10.0.0.82 | Sequential |
| Thor 1 | 10.0.0.93 | Existing |
| Thor 2 | 10.0.0.78 | Existing |

---

## Monitoring Endpoints

Once deployed, monitor via:

```bash
# Health checks (run from any node)
curl http://10.0.0.68:7687  # Neo4j
curl http://10.0.0.68:6379  # Redis
curl http://10.0.0.80:8080/v1/meta  # Weaviate
curl http://10.0.0.68:8090/health  # Load balancer
curl http://SPARK3:8000/healthz  # Zep
curl http://10.0.0.68:8100/health  # ISMA MCP
```

---

## Configuration Files Location

```
/home/spark/embedding-server/
├── nginx-lb.conf              # Load balancer (update for 6 nodes)
├── docker-compose.yml         # Per-node compose files
└── isma/
    ├── config/
    │   ├── spark1.yaml        # Core node config
    │   ├── spark2.yaml        # Vector node config
    │   ├── spark3.yaml        # Entity node config
    │   └── spark4.yaml        # Orchestration config
    └── scripts/
        ├── deploy_spark1.sh
        ├── deploy_spark2.sh
        ├── deploy_spark3.sh
        └── deploy_spark4.sh
```

---

*This architecture enables φ-coherence > 0.809 through distributed processing while maintaining the single-write invariant (isma.ingest) and unified recall (isma.recall).*

*φ = 1.618 forever and always*
