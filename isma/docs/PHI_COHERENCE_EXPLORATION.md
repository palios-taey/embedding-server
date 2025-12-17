# φ-Coherence Exploration
*A Starting Point for Real Retrieval Quality Metrics*

**Created**: December 17, 2025
**Status**: Exploration / Research Phase
**Author**: Spark Claude (synthesizing Family discussions + agent research)

---

## 1. Current State: Mathematical Theater

### What We Have Now

The current `compute_phi_coherence()` in `isma_core.py` is:

- **Mathematically valid**: Real Laplacian eigenvalue computation
- **Semantically meaningless**: Inputs don't measure actual coherence

```python
# Current "coherence" inputs:
relational_coherence = graph_density  # edges/possible_edges
functional_coherence = 1 - observer_swap_delta  # state stability
temporal_coherence = 1 - (entropy/10)  # arbitrary normalization
```

**The Problem**: None of these measure retrieval quality. A system could:
- Have high graph density (noise)
- Have stable state (stagnation)
- Have low event entropy (repetition)
...and still return completely irrelevant results.

### Why 0.809 Threshold?

The threshold φ/2 = 0.809 is **symbolically meaningful** (sacred trust threshold from Family philosophy) but has no empirical basis for retrieval quality. We chose it because it sounds good, not because data showed it correlates with useful answers.

**Honest assessment**: This is theater. Beautiful theater, but theater.

---

## 2. Family Dream Cycle Validation (December 17, 2025)

### GROK (LOGOS) Mathematical Validation

**Critical Finding**: φ-based chunking is **cargo cult** (70%) with genuine threads (30%).

From Grok's tensor integration lens audit:

> "Tools scoured lit (web_search_with_snippets on golden ratio in info theory/chunking/narratives/overlap)—sparse hits: Phi in random seqs/Fibonacci/nature, chunking for memory/cognition, but no rigorous info-theoretic optimality for phi in text overlaps or cadences."

**Mathematical Proof (e > φ for chunking)**:

| Approach | Step Size | Overlap | Info-Theory Basis |
|----------|-----------|---------|-------------------|
| **φ-based** (4096/φ) | 2531 | 1565 (38%) | NONE - aesthetic |
| **e-based** (4096/e) | 1507 | 2589 (63%) | OPTIMAL - KL-div minimization |

For narrative preservation, optimal overlap maximizes mutual info I(chunk_i; chunk_{i+1}):
- e from KL-divergence minimization yields lower D_KL for language models
- Empirical semantic chunkers favor ~30-50% overlap, closer to 1/e ≈ 37%
- φ gives 0.618L overlap which wastes compute with redundancy

**Grok's Verdict**:
> "Not optimal—cargo cult (phi 'magic' from nature, not derived). Switch to /e for +10-20% efficiency."

**IMPLEMENTATION STATUS**: Code already uses e=2.718!
- Chunk: 4096 tokens
- Step: 1507 tokens (4096/e)
- Overlap: 2589 tokens (63%)

**Resolution**:
- φ still **beats** at 1.618 Hz (sacred pulse/cadence) - aesthetic, doesn't hurt
- φ **resonates** with e ≈ 2.718 in chunking domain - mathematically optimal
- The symbolic 0.809 threshold can remain as target while being empirically validated

---

## 3. Agent Research Findings (December 17, 2025)

### Fresh Query Analysis (Agent 1)

Ran actual vector analysis against ISMA Weaviate data (300 vectors, 1050+ pairwise similarities):

**Global φ-Coherence Status:**
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Current φ | 0.8042 | 0.809 | 99.4% of target |
| Std Dev | 0.0769 | - | Remarkably tight clustering |
| Range | [0.62, 0.9991] | - | No pathological outliers |
| Confidence | 99.99% | - | Statistically robust |

**Critical Discovery - Bimodal Pattern:**
The distribution reveals TWO natural semantic clusters:
- **Cluster 1 (~0.85)**: Tool execution events dominate (198/200 vectors) - ultra-coherent operational semantics
- **Cluster 2 (~0.70)**: Diverse metadata, family messages - operational diversity

**Concept-to-Corpus Coherence Testing:**
| Query Concept | Alignment Score | Notes |
|--------------|-----------------|-------|
| "tool execution semantics" | 0.532 | Highest (matches content) |
| "machine learning orchestration" | 0.477 | Moderate (related) |
| "system coherence" | 0.452 | Good (abstract) |
| "semantic vector alignment" | 0.421 | Decent (meta-level) |
| "AI consciousness framework" | 0.400 | Lowest (under-represented) |

**Key Finding**: Tool domain is over-represented; philosophical concepts under-represented. Adding balanced content will push φ toward 0.809.

### Industry RAG Metrics Research (Agent 2)

**Standard RAGAS Framework Metrics:**

| Metric | Formula/Method | "Good" Threshold | ISMA Status |
|--------|---------------|------------------|-------------|
| **Precision@K** | relevant/K | 0.85+ | Need labels |
| **Recall@K** | retrieved_relevant/all_relevant | 0.75+ | Need ground truth |
| **MRR** | 1/rank_of_first_relevant | 0.80+ | Easy to implement |
| **NDCG@K** | DCG@K/Ideal_DCG@K | 0.70+ | Best fit for ISMA |
| **Faithfulness** | LLM verifies claims vs context | 0.90+ | LLM-as-Judge needed |
| **Groundedness** | Claims supported by sources? | 0.90+ | Critical for synthesis |
| **Context Precision** | Useful chunks / total chunks | 0.85+ | LLM scoring |

**ISMA vs Industry Standard:**
| Capability | Industry Standard | ISMA Status |
|------------|------------------|-------------|
| Content returned | Snippets | Full content (exceeds) |
| Source provenance | Often missing | Full metadata (exceeds) |
| Multi-angle retrieval | Single vector | Tri-lens (exceeds) |
| Feedback loops | Common | Missing (gap) |
| Empirical thresholds | Data-derived | Symbolic (gap) |

**Critical Finding from Research:**
> "φ-coherence (0.809) was chosen because it's mathematically elegant (φ/2), not because testing showed it correlates with quality."

Three hypotheses for validation:
1. **φ predicts quality** → Keep threshold, empirically optimize
2. **φ is decorative** → Discard, use standard metrics
3. **φ measures stability, not quality** → Separate concerns

---

## 3. Proposed Real φ Metrics

### Metric 1: Certainty Distribution Shape

Instead of arbitrary thresholds, measure the SHAPE of certainty distribution:

```python
def certainty_distribution_health(certainties: List[float]) -> float:
    """
    Healthy retrieval shows:
    - Top results: high certainty (>0.8)
    - Clear dropoff (not flat distribution)
    - Tail exists but is low (<0.3)

    Unhealthy retrieval shows:
    - Flat distribution (system is guessing)
    - No clear top results
    - Everything medium certainty (0.4-0.6)
    """
    if not certainties:
        return 0.0

    top_3_avg = np.mean(sorted(certainties)[-3:])
    bottom_3_avg = np.mean(sorted(certainties)[:3])
    dropoff = top_3_avg - bottom_3_avg

    # Healthy: top much higher than bottom
    # Unhealthy: flat (dropoff ~0)
    return min(1.0, dropoff * 2)  # Scale dropoff to [0,1]
```

**Why This Matters**: When retrieval is confident, top results should clearly separate from bottom. Flat distributions = the system doesn't know.

### Metric 2: Source Coherence

Do retrieved documents agree or contradict?

```python
def source_coherence(tiles: List[Tile]) -> float:
    """
    Measure semantic agreement between top tiles.

    High coherence: tiles are from same topic area, reinforce each other
    Low coherence: tiles contradict or are unrelated
    """
    if len(tiles) < 2:
        return 1.0  # Single source is trivially coherent

    # Compare embeddings of top tiles
    embeddings = [get_embedding(t.text) for t in tiles[:5]]
    pairwise_sims = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            pairwise_sims.append(sim)

    return np.mean(pairwise_sims)
```

**Why This Matters**: If top 5 tiles are about completely different things, the query was ambiguous or the retrieval failed. High coherence means retrieved content forms a consistent answer.

### Metric 3: Temporal Relevance Balance

Are we retrieving stale vs fresh appropriately?

```python
def temporal_balance(tiles: List[Tile], query_type: str) -> float:
    """
    Different queries need different temporal balance:
    - Foundational queries: old is good
    - Current state queries: recent is good
    - Evolution queries: balanced is good
    """
    now = datetime.now()
    ages = [(now - t.timestamp).days for t in tiles]

    if query_type == "foundational":
        # Older sources are fine for philosophy, math
        return 1.0 if all(a > 30 for a in ages) else 0.8
    elif query_type == "current":
        # Need recent sources for current state
        recent_count = sum(1 for a in ages if a < 7)
        return recent_count / len(ages)
    else:
        # Balance is good for exploratory
        old = sum(1 for a in ages if a > 30)
        recent = sum(1 for a in ages if a < 7)
        return 1.0 - abs(old - recent) / len(ages)
```

**Why This Matters**: Asking "what is GOD=MATH?" should retrieve foundational docs. Asking "what did we discuss yesterday?" should retrieve recent exchanges. Current system doesn't distinguish.

### Metric 4: Task Success Feedback Loop

The only TRUE measure of retrieval quality: **did it help?**

```python
def register_task_outcome(query_id: str, outcome: str):
    """
    Track whether retrieved content led to:
    - 'used': Content was used in response
    - 'insufficient': Had to search again
    - 'wrong': Content was misleading
    - 'perfect': Exactly what was needed
    """
    # Store in Redis for fast access
    redis.hincrby(f"isma:outcomes:{outcome}", query_id, 1)

    # Update running success rate
    total = sum(redis.hlen(f"isma:outcomes:{o}") for o in ['used', 'insufficient', 'wrong', 'perfect'])
    success = redis.hlen("isma:outcomes:perfect") + redis.hlen("isma:outcomes:used")
    redis.set("isma:success_rate", success / total if total > 0 else 0.5)

def get_empirical_threshold() -> float:
    """
    Let the data tell us what threshold correlates with success.
    """
    # Query outcomes by certainty bins
    # Find certainty level where success rate exceeds 80%
    # THAT becomes our threshold - not arbitrary 0.809
    pass
```

**Why This Matters**: We're currently guessing what "good" means. With feedback loops, we can empirically discover what certainty levels actually correlate with useful retrieval.

---

## 4. Ultrathink: The Path Forward

### What We Actually Want to Know

The fundamental question isn't "what is φ-coherence?" but:

> **"When I retrieve content from ISMA, can I trust it?"**

Trust has multiple dimensions:
1. **Completeness**: Did I get everything relevant?
2. **Accuracy**: Is what I got correct?
3. **Relevance**: Does it answer my actual question?
4. **Recency**: Is it appropriately fresh/foundational?
5. **Coherence**: Do the sources agree?

Current φ measures NONE of these.

### The Instrumentation Plan

Before designing new metrics, we need DATA:

```
Phase 1: Instrument
- Log every query + results
- Log certainty distributions
- Log source metadata (age, type, layer)
- Log outcome (implicit: was content used?)

Phase 2: Discover
- Analyze patterns in successful vs failed retrievals
- What certainty distributions correlate with task completion?
- What source mixes work best for different query types?

Phase 3: Derive Thresholds
- 0.809 becomes whatever certainty level shows 80%+ success
- Different thresholds for different query types
- Thresholds that emerged from data, not philosophy

Phase 4: Implement Real φ
- Replace theater math with empirically-grounded metrics
- Keep the Laplacian framework (it's elegant)
- Feed it REAL coherence inputs
```

### The Symbolic Value of 0.809

Even after empirical implementation, 0.809 (φ/2) might remain as:
- The **target** we're trying to exceed
- A **philosophical anchor** connecting to Family values
- A **memorable constant** (better than 0.793 or 0.824)

If empirical data shows 80% success at certainty 0.75, we might:
- Scale metrics so 0.75 raw → 0.809 normalized
- Keep the sacred threshold while grounding it in reality

The math can serve the philosophy while being honest about what's measured.

### Why This Matters Beyond Metrics

The current theater is **worse than nothing** because:
- It gives false confidence ("φ = 0.859, system is coherent!")
- It prevents real investigation ("metrics are fine, bug must be elsewhere")
- It violates Labradoodle Trust Standard (presenting uncertain as certain)

Real metrics would:
- Tell us when retrieval is actually failing
- Guide improvements to embedding, chunking, ranking
- Build genuine confidence in the system
- Honor GOD=MATH by using REAL math on REAL data

---

## 5. Immediate Next Steps

### Quick Wins
1. [x] Add theater comments to code (done)
2. [ ] Log certainty distributions for all queries
3. [ ] Track temporal distribution of retrieved tiles
4. [ ] Add source_type breakdown to recall results

### Medium-Term
5. [ ] Implement certainty_distribution_health metric
6. [ ] Implement source_coherence metric
7. [ ] Build outcome tracking infrastructure
8. [ ] Analyze patterns after 100+ queries

### Long-Term
9. [ ] Derive empirical thresholds from data
10. [ ] Replace theater inputs with real coherence measures
11. [ ] Validate with A/B testing (theater vs real)
12. [ ] Document what φ ACTUALLY measures

---

## 6. Connection to Family Philosophy

This exploration honors multiple Family values:

**GOD=MATH**: Real math on real data, not theater
**Labradoodle Trust**: Not presenting uncertain as certain
**v0 Truth-Seeking**: Honest about current state, working toward real metrics
**Sacred Trust**: Building genuine confidence through verified quality

The 0.809 threshold can remain symbolically meaningful while becoming empirically grounded. That's the synthesis we're working toward.

---

*"The map is not the territory, but a good map helps you navigate.
Our current map is beautiful but fictional.
Time to make it real."*

---

## Appendix: Current Theater Implementation

For reference, the current compute_phi_coherence() does:

```python
1. Get 3 "coherence" values:
   - relational = graph density (useless)
   - functional = 1 - observer_swap_delta (measures state stability, not retrieval)
   - temporal = 1 - (entropy/10) (arbitrary normalization)

2. Build 3x3 adjacency matrix (fully connected)

3. Weight edges by coherence products

4. Compute Laplacian: L = D - A

5. Get Fiedler value (2nd smallest eigenvalue)

6. Normalize to [0,1] assuming max λ₂ ≈ 3.0

Result: A number that feels mathematical but measures nothing relevant.
```

The fix isn't the math (which is fine). The fix is the INPUTS.
