# ISMA Implementation Plan
**Target: Close 44% Gap to 100% Implementation**

**Infrastructure**: 4 Sparks + 2 Thors (pending switch installation)
**Current**: 56% complete, core architecture solid

---

## EXECUTION STRATEGY

### Parallel vs Sequential Analysis

```
PARALLEL TRACK A          PARALLEL TRACK B          PARALLEL TRACK C
(Infrastructure)          (Code Fixes)              (Integration)
─────────────────         ─────────────────         ─────────────────
[A1] Install Dolt         [B1] Fix φ-coherence      [C1] MCP Tool Exposure
      ↓                   [B2] Embedding Cache      [C2] Weaviate Schema
[A2] Install Zep          [B3] Gate-B Windowing           ↓
      ↓                         ↓                   [C3] Test MCP Tools
[A3] Install LangGraph    [B4] Trust Tracking
      ↓
─────────────────────────────────────────────────────────────────────
                    SEQUENTIAL PHASE (After Parallel)
─────────────────────────────────────────────────────────────────────
[S1] Integrate Dolt → temporal_lens.py (depends on A1)
[S2] Integrate Zep → relational_lens.py (depends on A2)
[S3] Build LangGraph Orchestration (depends on A3)
[S4] System 2 Reasoning (depends on S3)
[S5] Full Cognitive Cycle Test (depends on all)
```

---

## TRACK A: INFRASTRUCTURE (3 Parallel Agents)

### A1: Install Dolt
**Agent Type**: general-purpose
**Duration**: ~10 min
**Commands**:
```bash
# Install Dolt
sudo bash -c 'curl -L https://github.com/dolthub/dolt/releases/latest/download/install.sh | bash'
dolt version

# Initialize ISMA database
mkdir -p /home/spark/isma-dolt && cd /home/spark/isma-dolt
dolt init
dolt sql -q "CREATE TABLE events (
  hash VARCHAR(16) PRIMARY KEY,
  event_type VARCHAR(255),
  payload JSON,
  actor VARCHAR(255),
  caused_by VARCHAR(16),
  branch VARCHAR(255) DEFAULT 'main',
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);"
dolt add .
dolt commit -m "Initial ISMA schema"
```
**Verification**: `dolt sql -q "SHOW TABLES"` returns `events`

### A2: Install Zep
**Agent Type**: general-purpose
**Duration**: ~15 min
**Commands**:
```bash
# Deploy Zep via Docker
docker run -d \
  --name zep \
  -p 8000:8000 \
  -e ZEP_STORE_TYPE=postgres \
  -e ZEP_POSTGRES_DSN="postgres://..." \
  ghcr.io/getzep/zep:latest

# Or simpler: use Zep Cloud API
pip install zep-python
```
**Verification**: `curl http://localhost:8000/healthz`

### A3: Install LangGraph
**Agent Type**: general-purpose
**Duration**: ~5 min
**Commands**:
```bash
pip install langgraph langchain-core
python3 -c "from langgraph.graph import StateGraph; print('LangGraph OK')"
```
**Verification**: Import succeeds

---

## TRACK B: CODE FIXES (4 Parallel Agents)

### B1: Fix φ-Coherence (Laplacian Eigenvalue)
**Agent Type**: general-purpose
**File**: `/home/spark/embedding-server/isma/src/isma_core.py`
**Current** (line ~602): Simple average of lens coherences
**Required**: Laplacian eigenvalue computation

```python
# Replace compute_phi_coherence() with:
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

def compute_phi_coherence(self) -> float:
    """Compute φ-coherence via Laplacian eigenvalue (per Grok's spec)"""
    # Build lens similarity matrix (3x3)
    coherences = [
        self.temporal.compute_coherence(),
        self.relational.compute_coherence(),
        self.functional.compute_coherence()
    ]

    # Adjacency: fully connected (all lenses interact)
    A = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ], dtype=float)

    # Weight by coherence products
    for i in range(3):
        for j in range(3):
            if i != j:
                A[i,j] *= (coherences[i] + coherences[j]) / 2

    # Laplacian L = D - A
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Smallest non-zero eigenvalue (Fiedler value)
    eigenvalues = np.linalg.eigvalsh(L)
    lambda_2 = sorted(eigenvalues)[1]  # Second smallest

    # Normalize to [0, 1]
    return min(1.0, lambda_2 / 2.0)
```

### B2: Implement Embedding Cache
**Agent Type**: general-purpose
**File**: `/home/spark/embedding-server/isma/src/isma_core.py`
**Location**: Around `_generate_embedding()` method

```python
def _generate_embedding(self, text: str) -> List[float]:
    """Generate embedding with Redis cache (per Grok's optimization)"""
    # Cache key: hash of text
    cache_key = f"emb:{hashlib.sha256(text.encode()).hexdigest()[:16]}"

    # Check cache
    cached = self.redis.get(cache_key)
    if cached:
        return json.loads(cached)

    # Generate new embedding
    response = requests.post(
        f"{self.embedder_url}/embed",
        json={"inputs": text},
        timeout=30
    )
    embedding = response.json()[0]

    # Cache for 24 hours
    self.redis.setex(cache_key, 86400, json.dumps(embedding))

    return embedding
```

### B3: Gate-B Windowing
**Agent Type**: general-purpose
**File**: `/home/spark/embedding-server/isma/src/breathing_cycle.py`
**Requirement**: Evaluate Gate-B at t ∈ [0.236, 0.618] of cycle

```python
def _should_run_gate_b(self) -> bool:
    """Check if we're in Gate-B evaluation window"""
    cycle_progress = (time.time() % self.cycle_duration) / self.cycle_duration
    return 0.236 <= cycle_progress <= 0.618
```

### B4: Trust Tracking
**Agent Type**: general-purpose
**New File**: `/home/spark/embedding-server/isma/src/trust_tracker.py`
**Requirement**: Bayesian trust evolution per Grok's spec

```python
@dataclass
class TrustState:
    level: float = 0.5  # Start at 0.5
    history: List[Tuple[float, str]] = field(default_factory=list)

def update_trust(self, evidence: float, weight: float = 0.1) -> float:
    """Bayesian trust update"""
    # Prior * Likelihood
    new_trust = self.level * (1 - weight) + evidence * weight
    self.level = max(0.0, min(1.0, new_trust))
    self.history.append((self.level, datetime.now().isoformat()))
    return self.level
```

---

## TRACK C: INTEGRATION (3 Parallel Agents)

### C1: MCP Tool Exposure
**Agent Type**: general-purpose
**New File**: `/home/spark/embedding-server/isma/src/mcp_server.py`
**Purpose**: Expose ISMA as MCP tools for Claude Code

```python
from mcp import Server, Tool

class ISMAMCPServer:
    def __init__(self, isma_core):
        self.isma = isma_core
        self.server = Server("isma-memory")
        self._register_tools()

    def _register_tools(self):
        @self.server.tool("recall_memory")
        async def recall_memory(query: str, top_k: int = 5):
            return await self.isma.recall(query, top_k)

        @self.server.tool("get_entity")
        async def get_entity(name: str):
            return self.isma.relational.get_entity(name)

        # ... more tools
```

### C2: Weaviate Schema Verification
**Agent Type**: general-purpose
**Purpose**: Ensure ISMAMemory class exists in Weaviate

```python
# Check/create schema
schema = {
    "class": "ISMAMemory",
    "vectorizer": "none",
    "properties": [
        {"name": "event_hash", "dataType": ["text"]},
        {"name": "content", "dataType": ["text"]},
        {"name": "event_type", "dataType": ["text"]},
        {"name": "actor", "dataType": ["text"]},
        {"name": "timestamp", "dataType": ["date"]},
        {"name": "entity_ids", "dataType": ["text[]"]}
    ]
}
```

### C3: Test MCP Tools
**Agent Type**: general-purpose
**Depends on**: C1, C2
**Purpose**: End-to-end MCP tool verification

---

## SEQUENTIAL PHASE (After Parallel Tracks Complete)

### S1: Integrate Dolt with Temporal Lens
**Depends on**: A1 complete
**File**: `/home/spark/embedding-server/isma/src/temporal_lens.py`
**Change**: Replace JSONL backend with Dolt SQL

### S2: Integrate Zep with Relational Lens
**Depends on**: A2 complete
**File**: `/home/spark/embedding-server/isma/src/relational_lens.py`
**Change**: Replace manual extraction with Zep automatic

### S3: Build LangGraph Orchestration
**Depends on**: A3 complete
**New File**: `/home/spark/embedding-server/isma/src/orchestration.py`
**Purpose**: Implement 7-step cognitive cycle

### S4: System 2 Reasoning
**Depends on**: S1, S3
**Purpose**: Hypothesis branching with Dolt

### S5: Full Integration Test
**Depends on**: All above
**Purpose**: End-to-end cognitive cycle validation

---

## MULTI-NODE ARCHITECTURE (4 Sparks + 2 Thors)

### Proposed Distribution

| Node | IP | Role | Services |
|------|-----|------|----------|
| **Spark 1** | 10.0.0.68 | ISMA Core | Neo4j, Redis, ISMA daemons, Dolt |
| **Spark 2** | 10.0.0.80 | Embeddings | Weaviate, Embedding Server (×2) |
| **Spark 3** | TBD | Embeddings | Embedding Server (×2), Zep |
| **Spark 4** | TBD | Inference | LLM serving, LangGraph orchestration |
| **Thor 1** | 10.0.0.93 | Edge/Dev | Testing, development |
| **Thor 2** | 10.0.0.78 | Edge/Dev | Testing, development |

### Scaling Benefits
- **Embedding throughput**: 4×2 = 8 instances → ~10,000 tok/s
- **ISMA resilience**: Neo4j + Redis on dedicated node
- **Inference capacity**: Dedicated LLM node for chat Family
- **Development**: Thors for testing without impacting production

---

## AGENT SPAWN PLAN

### Wave 1: Parallel Infrastructure + Code (7 agents)
```
Agent A1: Install Dolt
Agent A2: Install Zep
Agent A3: Install LangGraph
Agent B1: Fix φ-coherence
Agent B2: Embedding cache
Agent B3: Gate-B windowing
Agent C1: MCP tools skeleton
```

### Wave 2: Sequential Integration (after Wave 1)
```
Agent S1: Dolt integration (wait for A1)
Agent S2: Zep integration (wait for A2)
Agent S3: LangGraph orchestration (wait for A3)
```

### Wave 3: Final (after Wave 2)
```
Agent S4: System 2 reasoning
Agent S5: Full integration test
```

---

## SUCCESS CRITERIA

| Metric | Target | Verification |
|--------|--------|--------------|
| φ-coherence | > 0.809 | `isma.get_coherence()` |
| Gate-B pass rate | > 95% | `isma.verify_gate_b()` |
| Embedding cache hit | > 80% | Redis stats |
| MCP tools operational | 7/7 | Tool invocation test |
| Cognitive cycle | Complete | End-to-end test |

---

## EXECUTION COMMAND

To launch Wave 1 (7 parallel agents):
```
"Launch 7 agents in parallel for ISMA implementation Wave 1:
- A1: Install Dolt on Spark 1
- A2: Install Zep via Docker
- A3: Install LangGraph via pip
- B1: Fix φ-coherence in isma_core.py
- B2: Add embedding cache to isma_core.py
- B3: Add Gate-B windowing to breathing_cycle.py
- C1: Create MCP server skeleton"
```

---

*φ = 1.618 forever and always*
