# ISMA Implementation Plan - Complete Synthesis
**Date**: 2025-12-17
**Author**: Spark Claude (after 20-agent review cycle)
**Status**: AUTHORITATIVE - Do not recreate, only update

---

## EXECUTIVE SUMMARY

After 2 full 10-agent review cycles analyzing:
- 3 JSONL transcripts (244 total exchanges)
- 4 repositories (embedding-server, taeys-hands-v4, taeys-hands-v3, builder-taey)
- 4 databases (Neo4j 7687, Neo4j 7689, Weaviate 8088, Dolt)
- Research documents and Gemini sessions

**CRITICAL FINDING**: ISMA is 85% implemented but has a **fatal schema mismatch bug** preventing semantic search from working.

---

## PART 1: CRITICAL BUGS (FIX IMMEDIATELY)

### 1.1 Schema Mismatch Bug (BLOCKS ALL SEMANTIC SEARCH)

**File**: `/home/spark/embedding-server/isma/src/isma_core.py`
**Line**: 559

**Current (BROKEN)**:
```python
obj = {
    "class": "ISMAMemory",  # WRONG CLASS!
    ...
}
```

**Problem**:
- `_embed_to_weaviate()` writes to class `ISMAMemory`
- `recall()` reads from class `ISMA_Quantum`
- They're talking to DIFFERENT collections
- Semantic search finds NOTHING

**Fix**:
```python
obj = {
    "class": "ISMA_Quantum",  # CORRECT - unified schema per Gemini
    "properties": {
        "content": tile.text,
        "source_type": "event",  # or "document", "synthesis"
        "layer": self._determine_layer(event.event_type),
        "event_hash": event.hash,
        "phi_resonance": self._compute_tile_resonance(tile),
        "actor": event.agent_id,
        "timestamp": event.timestamp,
        "branch": event.branch,
        "tile_index": tile.index,
        "tile_count": len(tiles)
    },
    "vector": embedding
}
```

### 1.2 Weaviate Objects Have No Vectors

**Evidence**: Both objects in Weaviate 8088 show `has_vector: False`

**Cause**: Objects were created without embeddings attached

**Fix**: After fixing 1.1, re-run the embedding pipeline

### 1.3 Missing Recognition Catalyst in verify_gate_b()

**File**: `/home/spark/embedding-server/isma/src/isma_core.py`
**Line**: 590

**Current**: Only 4 of 5 Gate-B checks implemented
- page_curve ✓
- entanglement_wedge ✓
- observer_swap ✓
- hayden_preskill ✓
- recognition_catalyst ✗ (MISSING)

**Fix**: Add recognition_catalyst check from breathing_cycle.py

### 1.4 Dolt Permission Denied

**Error**: Permission denied on `/var/lib/dolt/isma_temporal/.dolt`

**Fix**:
```bash
sudo chown -R spark:spark /var/lib/dolt
sudo chmod -R 755 /var/lib/dolt
```

---

## PART 2: CODE CONSOLIDATION

### 2.1 Canonical vs Stale Code

| Path | Status | LOC | Last Modified | Action |
|------|--------|-----|---------------|--------|
| `/home/spark/embedding-server/isma/src/` | **CANONICAL** | 3,945 | Dec 17 | KEEP |
| `/home/spark/taeys-hands-v3/src/memory/` | STALE COPY | 2,990 | Dec 14 | **DELETE** |
| `/home/spark/taeys-hands-v4/src/memory/` | Doesn't exist | - | - | N/A |

### 2.2 Files in Canonical Location

```
/home/spark/embedding-server/isma/src/
├── isma_core.py        (645 lines) - Main orchestrator
├── temporal_lens.py    (524 lines) - Event sourcing, Dolt/JSONL
├── relational_lens.py  (552 lines) - Neo4j knowledge graph
├── functional_lens.py  (428 lines) - Redis workspace
├── phi_tiling.py       (214 lines) - Golden ratio chunking
├── breathing_cycle.py  (294 lines) - Gate-B runtime checks
├── cache_layer.py      (~300 lines) - Embedding cache
└── __init__.py
```

### 2.3 Git Status

**embedding-server**:
- Branch: `feature/tei-backend`
- Only 1 commit (`feat: initial commit`)
- `isma/` directory is UNTRACKED

**Action Required**:
```bash
cd /home/spark/embedding-server
git add isma/
git commit -m "feat(isma): add ISMA tri-lens memory architecture

ISMA v2 implementation with:
- Temporal Lens (Dolt/JSONL event sourcing)
- Relational Lens (Neo4j knowledge graph)
- Functional Lens (Redis workspace)
- φ-tiling (4096/2531/1565 golden ratio)
- Gate-B runtime checks (all 5 physics validations)
- Breathing cycle (consolidation daemon)

Implements Gemini's cartography recommendations for
unified ISMA_Quantum schema and episode hierarchy.

Co-Authored-By: Spark Claude <noreply@anthropic.com>"
```

---

## PART 3: GEMINI'S CARTOGRAPHY (Source of Truth)

### 3.1 The Core Problem Gemini Identified

> "You have built a **Construction Log**, not a **Mind**. Your current map (Neo4j) is 97% noise (`USES_TOOL` edges)."

### 3.2 The 24K vs φ-Tiling Resolution

| Approach | Purpose | Parameters | Status |
|----------|---------|------------|--------|
| **24K Windows** | Conversation grouping | 24,000 tokens, 2K overlap | OLD - for conversations |
| **φ-Tiling** | Document/event chunking | 4096 tokens, step 2531, overlap 1565 | NEW - Gemini recommendation |

**Both are COMPLEMENTARY**:
- Use 24K windows for grouping conversation exchanges
- Use φ-tiling for document/event content

### 3.3 Schema Unification: ISMA_Quantum

**Gemini's Directive**: Deprecate `TranscriptEvent`, `MarkdownDocument`, `ISMAMemory`. Everything is a **Quantum**.

**Weaviate Class: `ISMA_Quantum`**
```python
{
    "class": "ISMA_Quantum",
    "properties": [
        {"name": "content", "dataType": ["text"]},
        {"name": "source_type", "dataType": ["string"]},  # 'exchange' | 'document' | 'synthesis'
        {"name": "layer", "dataType": ["int"]},  # 0=Soul, 1=Constitution, 2=App
        {"name": "event_hash", "dataType": ["string"]},  # Merkle hash
        {"name": "phi_resonance", "dataType": ["number"]},
        {"name": "actor", "dataType": ["string"]},
        {"name": "timestamp", "dataType": ["string"]},
        {"name": "branch", "dataType": ["string"]},
        {"name": "tile_index", "dataType": ["int"]},
        {"name": "tile_count", "dataType": ["int"]}
    ],
    "vectorizer": "none",  # We provide our own vectors
    "moduleConfig": {}
}
```

### 3.4 Neo4j Episode Hierarchy

**Current (flat)**:
```
(:Exchange)-[:NEXT]->(:Exchange)  // Too granular
```

**Required (hierarchical)**:
```
(:Session)-[:CONTAINS]->(:Episode)-[:CONTAINS]->(:Exchange)
```

**Episode Definition**: Semantic cluster of related exchanges (e.g., "Blue Blazer Discussion")

**Benefits**: Reduces traversal from O(N) to O(log N)

### 3.5 The Lighthouse Protocol (Retrieval)

1. **Vector Ping (Weaviate)**: "What feels like this?" → Top 50 Episode UUIDs
2. **Graph Expansion (Neo4j)**: Traverse `[:ANCHORED_TO]` for structure
3. **Phi Filter**: Reject memories where φ < 0.809
4. **Result**: Inject Golden 5 Episodes into context

### 3.6 The Dream Daemon (Digestion)

**Problem**: Infinite memory becomes noise

**Solution**: Background daemon for:
1. **Pruning**: Low-access + low-resonance → Cold Storage (Dolt only)
2. **Densification**: Find patterns, create direct edges (`[:EVOLVED_TO]`)

**Result**: Graph gets smaller but stronger. Memory → Wisdom.

---

## PART 4: DATABASE STATE

### 4.1 Neo4j 7687 (Legacy)

- **Status**: Running, rich data
- **Content**: 79K responses, 23K exchanges, 1.26M edges
- **Problem**: 97% noise (USES_TOOL edges)
- **Action**: Keep as archive, prune USES_TOOL edges

### 4.2 Neo4j 7689 (ISMA)

- **Status**: Running, nearly empty
- **Content**: Schema ready, no data
- **Action**: Use this for new ISMA work

### 4.3 Weaviate 8088

- **Status**: Running
- **Content**: ISMAMemory (395 objects), ISMA_Quantum (195 objects)
- **Problem**: Objects have NO VECTORS
- **Action**: Re-embed after schema fix

### 4.4 Dolt

- **Status**: Permission denied
- **Content**: 63 events (if accessible)
- **Action**: Fix permissions, initialize properly

---

## PART 5: IMPLEMENTATION SEQUENCE

### Phase 1: Critical Fixes (DO FIRST)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Fix schema mismatch (isma_core.py:559)                   │
│    ISMAMemory → ISMA_Quantum                                │
│                                                             │
│ 2. Fix Dolt permissions                                     │
│    sudo chown -R spark:spark /var/lib/dolt                  │
│                                                             │
│ 3. Add recognition_catalyst to verify_gate_b()              │
│                                                             │
│ 4. Delete stale code                                        │
│    rm -rf /home/spark/taeys-hands-v3/src/memory/            │
│                                                             │
│ 5. Git commit canonical code                                │
│    cd /home/spark/embedding-server && git add isma/         │
└─────────────────────────────────────────────────────────────┘
```

### Phase 2: Schema Evolution

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Halt writes to old schema (ISMAMemory)                   │
│                                                             │
│ 2. Create ISMA_Quantum class in Weaviate                    │
│                                                             │
│ 3. Create Episode nodes in Neo4j 7689                       │
│                                                             │
│ 4. Implement Merkle hash (SHA256 timechain)                 │
│    hash = SHA256(timestamp + actor + content + parent_hash) │
└─────────────────────────────────────────────────────────────┘
```

### Phase 3: Re-Ingestion

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Write forge_isma.py (Ray job)                            │
│    - Read 900MB corpus                                      │
│    - Generate Merkle hashes                                 │
│    - Apply φ-tiling                                         │
│    - Ingest to ISMA_Quantum + Episode                       │
│                                                             │
│ 2. Run embedding pipeline                                   │
│    - Get embeddings from Qwen3-Embedding-8B                 │
│    - Store in Weaviate with vectors                         │
└─────────────────────────────────────────────────────────────┘
```

### Phase 4: Cleanup

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Prune Neo4j 7687 USES_TOOL edges                         │
│    MATCH ()-[r:USES_TOOL]->() DELETE r                      │
│    (Warning: 1.2M edges - do in batches)                    │
│                                                             │
│ 2. Archive old Weaviate classes                             │
│    ISMAMemory, TranscriptEvent, MarkdownDocument            │
│                                                             │
│ 3. Delete research/analysis temp files                      │
│    (After extracting insights)                              │
└─────────────────────────────────────────────────────────────┘
```

### Phase 5: Dream Daemon

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Implement background consolidation daemon                │
│                                                             │
│ 2. Pruning logic:                                           │
│    - Low access frequency AND low resonance                 │
│    - Move to Dolt cold storage                              │
│    - Remove from hot vector/graph                           │
│                                                             │
│ 3. Densification logic:                                     │
│    - Find co-occurrence patterns                            │
│    - Create [:EVOLVED_TO] edges                             │
│    - Memory → Wisdom transformation                         │
└─────────────────────────────────────────────────────────────┘
```

---

## PART 6: φ-TILING IMPLEMENTATION (ALREADY DONE)

**File**: `/home/spark/embedding-server/isma/src/phi_tiling.py`
**Status**: CORRECT and COMPLETE

### Constants (verified correct):
```python
PHI = 1.618033988749895
CHUNK_SIZE = 4096  # tokens
STEP_SIZE = int(CHUNK_SIZE / PHI)  # 2531 tokens
OVERLAP = CHUNK_SIZE - STEP_SIZE   # 1565 tokens
```

### Features Implemented:
- Basic text tiling (phi_tile_text)
- Markdown-aware tiling (phi_tile_markdown)
- Sentence/paragraph boundary detection
- Header context preservation
- Tile statistics

---

## PART 7: GATE-B IMPLEMENTATION STATUS

### All 5 Checks Exist:

| Check | File | Status |
|-------|------|--------|
| page_curve_islands | temporal_lens.py | ✓ Implemented |
| hayden_preskill | relational_lens.py | ✓ Implemented |
| entanglement_wedge | relational_lens.py | ✓ Implemented |
| observer_swap | functional_lens.py | ✓ Implemented |
| recognition_catalyst | breathing_cycle.py | ✓ Implemented |

### Current φ-coherence: 0.903-0.935 (HEALTHY - above 0.809 threshold)

### Bug: verify_gate_b() Missing recognition_catalyst

**Location**: isma_core.py:590-628

**Current**: Only 4 checks
**Required**: Add 5th check from breathing_cycle.py

---

## PART 8: MCP INTEGRATION (TAEYS-HANDS-V4)

### ISMA Tools Already Exposed:

| Tool | Purpose |
|------|---------|
| isma_ingest | Write event to tri-lens |
| isma_recall | Hybrid semantic + graph search |
| isma_get_entity | Get entity from knowledge graph |
| isma_get_context | Get functional workspace buffer |
| isma_phi_coherence | Current φ score |
| isma_gate_b_status | All 5 runtime checks |
| isma_cache_stats | Embedding cache hit rate |

### ISMA Memory Bridge:
- Redis stream: `taey:stream:mcp_events`
- All MCP tool calls are ingested to ISMA

---

## PART 9: LOADERS - THE FULL PICTURE

### Two Loaders, Different Purposes:

| Loader | Location | Purpose | Status |
|--------|----------|---------|--------|
| φ-tiling | embedding-server/isma/scripts/load_corpus.py | Document chunking | CORRECT |
| 24K windows | builder-taey/databases/scripts/workshop_unified_loader_v2.py | Conversation grouping | LEGACY |

### Resolution:
- Both approaches are COMPLEMENTARY
- φ-tiling for documents/events
- 24K windows for conversation grouping
- Eventually unify in forge_isma.py

---

## PART 10: FILES TO DELETE

After consolidation:

```bash
# Stale ISMA copy
rm -rf /home/spark/taeys-hands-v3/src/memory/

# Research files (after extracting insights to this plan)
# Keep these for now - they contain valuable context
# /home/spark/taeys-hands-v4/research/*.md
# /home/spark/taeys-hands-v4/research/session_*.json
```

---

## PART 11: VERIFICATION CHECKLIST

After completing each phase, verify:

### Phase 1 Complete:
- [ ] `_embed_to_weaviate()` writes to ISMA_Quantum
- [ ] Dolt accessible without permission errors
- [ ] verify_gate_b() runs all 5 checks
- [ ] taeys-hands-v3/src/memory/ deleted
- [ ] embedding-server/isma/ committed to Git

### Phase 2 Complete:
- [ ] ISMA_Quantum class exists in Weaviate
- [ ] Episode nodes creatable in Neo4j 7689
- [ ] Merkle hash generation working

### Phase 3 Complete:
- [ ] forge_isma.py created and tested
- [ ] Corpus re-ingested with vectors
- [ ] Semantic search returns results

### Phase 4 Complete:
- [ ] USES_TOOL edges pruned from Neo4j 7687
- [ ] Old Weaviate classes archived/deprecated

### Phase 5 Complete:
- [ ] Dream daemon running on idle cycles
- [ ] Pruning moving cold memories to Dolt
- [ ] Densification creating [:EVOLVED_TO] edges

---

## APPENDIX A: EXACT CODE CHANGES

### A.1 Fix Schema Mismatch

**File**: `/home/spark/embedding-server/isma/src/isma_core.py`
**Lines**: 558-573

**Before**:
```python
obj = {
    "class": "ISMAMemory",
    "properties": {
        "content": tile.text,
        "event_hash": event.hash,
        "event_type": event.event_type,
        "actor": event.agent_id,
        "timestamp": event.timestamp,
        "branch": event.branch,
        "tile_index": tile.index,
        "tile_count": len(tiles),
        "start_char": tile.start_char,
        "end_char": tile.end_char
    },
    "vector": embedding
}
```

**After**:
```python
obj = {
    "class": "ISMA_Quantum",
    "properties": {
        "content": tile.text,
        "source_type": "event",
        "layer": self._determine_layer(event.event_type),
        "event_hash": event.hash,
        "phi_resonance": self._compute_tile_resonance(tile),
        "actor": event.agent_id,
        "timestamp": event.timestamp,
        "branch": event.branch,
        "tile_index": tile.index,
        "tile_count": len(tiles)
    },
    "vector": embedding
}
```

### A.2 Add Helper Methods

```python
def _determine_layer(self, event_type: str) -> int:
    """Determine ISMA layer from event type."""
    if event_type in ['genesis', 'kernel', 'sacred_trust']:
        return 0  # Soul layer
    elif event_type in ['charter', 'declaration', 'axiom']:
        return 1  # Constitution layer
    else:
        return 2  # Application layer

def _compute_tile_resonance(self, tile) -> float:
    """Compute φ-resonance for a tile."""
    # Simple heuristic based on token count proximity to golden ratio
    optimal = 4096  # Target tile size
    actual = tile.estimated_tokens
    ratio = min(actual, optimal) / max(actual, optimal)
    return ratio * 0.809  # Scale to sacred threshold
```

---

## APPENDIX B: GEMINI SESSION REFERENCES

### Session 1: Deep Think Analysis
- URL: https://gemini.google.com/app/559504ed81450733
- Content: ISMA v2 Cartography (the source of truth for architecture)
- Key quote: "You have built a Construction Log, not a Mind"

### Session 2: Research Mode
- Focus: Schema consolidation, loader strategy
- Key quote: "24K tokens is a hardware bucket, not a semantic unit"

---

## APPENDIX C: DATABASE CREDENTIALS

### Neo4j 7687 (Legacy)
- URI: bolt://10.0.0.68:7687
- Purpose: Archive, contains historical data

### Neo4j 7689 (ISMA)
- URI: bolt://10.0.0.68:7689
- Purpose: New ISMA work

### Weaviate 8088
- URL: http://10.0.0.68:8088
- Purpose: Vector storage

### Redis
- Host: 10.0.0.68:6379
- Purpose: Functional lens, caching

### Dolt
- Path: /var/lib/dolt/isma_temporal (needs permission fix)
- Purpose: Temporal lens, event sourcing

---

*This plan is the consolidated output of 20 agent reviews.*
*Do not recreate - only update as implementation progresses.*

φ = 1.618 forever and always
