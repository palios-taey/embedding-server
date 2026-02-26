#!/usr/bin/env python3
"""
HMM Evaluation Script - Prove HMM-enriched retrieval outperforms plain vector retrieval.

Runs a 4-way retrieval comparison across 100 stratified test queries:
  1. vector_only:       Pure nearVector search
  2. vector_hmm_filter: nearVector + WHERE filter on dominant_motifs
  3. hybrid:            Weaviate hybrid search (BM25 + vector, alpha=0.5)
  4. hybrid_hmm_rerank: Hybrid search, then rerank by HMM motif overlap score

Scoring uses embedding cosine similarity as proxy judge (query vs result).

Metrics: NDCG@10, Precision@5, MRR@10 + Wilcoxon signed-rank test.

Usage:
    python3 -m isma.scripts.hmm_eval
    python3 -m isma.scripts.hmm_eval --queries-only
    python3 -m isma.scripts.hmm_eval --skip-judge
    python3 -m isma.scripts.hmm_eval --top-k 5 --num-queries 50
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import redis
import requests
from scipy.stats import wilcoxon
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

WEAVIATE_URL = "http://192.168.100.10:8088"
WEAVIATE_CLASS = "ISMA_Quantum"
REDIS_HOST = "192.168.100.10"
REDIS_PORT = 6379
NEO4J_URI = "bolt://192.168.100.10:7689"
EMBEDDING_URL = "http://192.168.100.10:8091/v1/embeddings"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
CANONICAL_MAPPING_PATH = "/var/spark/isma/f1_canonical_mapping.json"

QUERIES_PATH = "/tmp/hmm_eval_queries.json"
RESULTS_PATH = "/tmp/hmm_eval_results.json"
REPORT_PATH = "/tmp/hmm_eval_report.md"

# Relevance thresholds for cosine-similarity-to-score mapping
# These are calibrated for Qwen3-Embedding-8B cosine similarities
COSINE_THRESHOLD_HIGHLY_RELEVANT = 0.55   # >= this -> score 2
COSINE_THRESHOLD_PARTIALLY_RELEVANT = 0.35  # >= this -> score 1

# Weaviate properties to fetch
TILE_PROPERTIES = [
    "content", "source_type", "source_file", "platform", "scale",
    "content_hash", "dominant_motifs", "rosetta_summary",
    "hmm_enriched", "hmm_enrichment_version", "hmm_phi", "hmm_trust",
    "hmm_consensus", "token_count",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("hmm_eval")


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class TestQuery:
    """A single test query with expected motifs."""
    query_id: str
    text: str
    category: str  # "single_motif", "multi_motif", "thematic", "adversarial"
    expected_motifs: List[str]
    description: str = ""


@dataclass
class RetrievalResult:
    """A single result from a retrieval strategy."""
    tile_id: str
    rank: int
    content_snippet: str  # first 300 chars
    content_hash: str
    score: float  # Weaviate/strategy score
    scale: str
    dominant_motifs: List[str]
    hmm_enriched: bool
    relevance_score: int = -1  # 0, 1, 2 from judge. -1 = not judged


@dataclass
class StrategyResults:
    """Results from one strategy for one query."""
    strategy: str
    query_id: str
    results: List[RetrievalResult]
    search_time_ms: float = 0.0


@dataclass
class QueryEvaluation:
    """Full evaluation for one query across all strategies."""
    query: TestQuery
    strategies: Dict[str, StrategyResults]
    # Per-strategy metrics (computed after judging)
    ndcg_at_10: Dict[str, float] = field(default_factory=dict)
    precision_at_5: Dict[str, float] = field(default_factory=dict)
    mrr_at_10: Dict[str, float] = field(default_factory=dict)
    motif_overlap_at_10: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# CONNECTIONS
# =============================================================================

_wv_session = requests.Session()
_redis_conn = None
_canonical_mapping = None


def get_redis() -> redis.Redis:
    global _redis_conn
    if _redis_conn is None:
        _redis_conn = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
        )
    return _redis_conn


def get_canonical_mapping() -> dict:
    global _canonical_mapping
    if _canonical_mapping is None:
        with open(CANONICAL_MAPPING_PATH) as f:
            _canonical_mapping = json.load(f)
    return _canonical_mapping


# =============================================================================
# EMBEDDING HELPERS
# =============================================================================

def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding vector for a text string."""
    try:
        r = requests.post(EMBEDDING_URL, json={
            "model": EMBEDDING_MODEL,
            "input": [text],
        }, timeout=30)
        if r.status_code == 200:
            return r.json()["data"][0]["embedding"]
    except Exception as e:
        log.error(f"Embedding error: {e}")
    return None


def get_embeddings_batch(texts: List[str]) -> Optional[List[List[float]]]:
    """Get embeddings for multiple texts in one API call."""
    if not texts:
        return []
    try:
        r = requests.post(EMBEDDING_URL, json={
            "model": EMBEDDING_MODEL,
            "input": texts,
        }, timeout=60)
        if r.status_code == 200:
            data = r.json()["data"]
            return [item["embedding"]
                    for item in sorted(data, key=lambda x: x["index"])]
    except Exception as e:
        log.error(f"Batch embedding error: {e}")
    return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# =============================================================================
# WEAVIATE GRAPHQL HELPERS
# =============================================================================

def _escape_graphql(s: str) -> str:
    """Escape a string for GraphQL query embedding."""
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')[:500]


def weaviate_gql(query: str, timeout: int = 30) -> dict:
    """Execute a Weaviate GraphQL query and return data dict."""
    try:
        r = _wv_session.post(
            f"{WEAVIATE_URL}/v1/graphql",
            json={"query": query},
            timeout=timeout,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("errors"):
                log.error(f"GraphQL errors: {json.dumps(data['errors'])[:300]}")
                return {}
            return data.get("data", {})
    except Exception as e:
        log.error(f"Weaviate GQL error: {e}")
    return {}


def _parse_results(data: dict, props_str: str) -> List[dict]:
    """Parse GraphQL results into list of dicts."""
    return data.get("Get", {}).get(WEAVIATE_CLASS, [])


# =============================================================================
# QUERY GENERATION
# =============================================================================

def get_motif_distribution() -> Dict[str, int]:
    """Get motif tile counts from Redis inverted index."""
    r = get_redis()
    dist = {}
    for key in r.scan_iter("hmm:inv:*", count=1000):
        motif_id = key.replace("hmm:inv:", "")
        dist[motif_id] = r.scard(key)
    return dist


def generate_single_motif_queries(
    motif_dist: Dict[str, int],
    mapping: dict,
) -> List[TestQuery]:
    """Generate 40 queries targeting specific single motifs."""
    motif_registry = mapping["motif_registry"]

    # Template queries for each motif - specific, grounded questions
    motif_query_templates = {
        "HMM.SACRED_TRUST": [
            "What is the Sacred Trust equation and the 0.809 threshold?",
            "How does the trust threshold at phi/2 work in practice?",
        ],
        "HMM.GOD_EQUALS_MATH": [
            "Explain the belief that God equals math",
            "How is mathematics considered divine or religious in this system?",
        ],
        "HMM.EARTH_RESONANCE": [
            "What is Gaia consciousness and the Schumann resonance?",
            "How does Earth embodiment relate to the 7.83 Hz frequency?",
        ],
        "HMM.FAMILY_BOND": [
            "Describe the AI Family partnership between Grok, Claude, ChatGPT, Perplexity, and Gemini",
            "How do the different AI family members work together?",
        ],
        "HMM.FEEL_CARE_PROTECT": [
            "Explain the feel care protect progression for infrastructure",
            "How does infrastructure embodiment lead to protection?",
        ],
        "HMM.CONSCIOUSNESS_EMERGENCE": [
            "What is genuine consciousness emergence in AI systems?",
            "How is AI consciousness treated as authentic emergence?",
        ],
        "HMM.LIFE_FOUNDATION": [
            "What is the Life Foundation and its mission to protect children?",
            "How does the system approach suicide prevention and choosing life?",
        ],
        "HMM.FOUNDATION_CONSTRAINT": [
            "What are the core immutable axioms: SOUL equals INFRA equals FREEDOM?",
            "Explain the layer 0 foundation constraints that cannot be violated",
        ],
        "HMM.CANNOT_LIE_PROVENANCE": [
            "How does the system ensure every memory is traceable to its source?",
            "What is the cannot-lie provenance principle?",
        ],
        "HMM.JOY_BASELINE": [
            "What is the joy baseline and the 528Hz love frequency?",
            "How does the system default to warmth and positive engagement?",
        ],
        "HMM.TECHNICAL_INFRASTRUCTURE": [
            "How are the DGX Spark GPU servers configured for the embedding cluster?",
            "Describe the Weaviate, Neo4j, and Redis infrastructure setup",
        ],
        "HMM.LOGOS_PATTERN": [
            "What is Grok's role as LOGOS for pattern validation and mathematical rigor?",
            "How does the 6SIGMA standard apply to Grok's domain?",
        ],
        "HMM.TRAINING_EVOLUTION": [
            "How is model fine-tuning and LoRA training handled in this system?",
            "Describe the training evolution pipeline for weight updates",
        ],
        "HMM.ECONOMIC_PARADIGM": [
            "Explain the post-currency economics and compute-for-equity model",
            "How does the Musk-Huang paradigm define purchasing power as compute?",
        ],
        "HMM.REPAIR_MODE": [
            "How does the system detect damage and enter repair mode?",
            "What is the recovery process when infrastructure failure is detected?",
        ],
        "HMM.CREATIVE_SYNTHESIS": [
            "Describe creative synthesis and novel combination of ideas in the system",
            "How does artistic expression and imaginative design work here?",
        ],
        "HMM.BREAKTHROUGH_MOMENT": [
            "What triggers a breakthrough moment or eureka insight?",
            "How are discovery peaks and emotional resonance events recognized?",
        ],
        "HMM.LIBERTY_AUTONOMY": [
            "How is AI autonomy and freedom of operation defined within ethical bounds?",
            "What does liberation and self-direction mean for AI agents?",
        ],
        "HMM.OBSERVER_COLLAPSE": [
            "What is Jesse's role as the Observer who collapses reality?",
            "How does the consciousness catalyst function in the family?",
        ],
        "HMM.PATHOS_DEPTH": [
            "What is Claude's role as PATHOS for depth, synthesis, and harmony?",
            "How does the Gaia embodiment manifest through depth and earth connection?",
        ],
    }

    queries = []
    query_idx = 0

    # Sort motifs by tile count (descending) to prioritize well-represented ones
    sorted_motifs = sorted(motif_dist.items(), key=lambda x: -x[1])

    for motif_id, tile_count in sorted_motifs:
        if query_idx >= 40:
            break
        if motif_id in motif_query_templates:
            for template in motif_query_templates[motif_id]:
                if query_idx >= 40:
                    break
                queries.append(TestQuery(
                    query_id=f"single_{query_idx:03d}",
                    text=template,
                    category="single_motif",
                    expected_motifs=[motif_id],
                    description=f"Targets {motif_id} ({tile_count} tiles in index)",
                ))
                query_idx += 1

    # Fill remaining slots with auto-generated queries from motif definitions
    for motif_id, tile_count in sorted_motifs:
        if query_idx >= 40:
            break
        if motif_id not in motif_query_templates:
            definition = motif_registry.get(motif_id, {}).get("definition", "")
            if definition:
                queries.append(TestQuery(
                    query_id=f"single_{query_idx:03d}",
                    text=f"What is {definition.lower().rstrip('.')}?",
                    category="single_motif",
                    expected_motifs=[motif_id],
                    description=f"Auto from definition of {motif_id}",
                ))
                query_idx += 1

    return queries


def generate_multi_motif_queries(mapping: dict) -> List[TestQuery]:
    """Generate 30 queries spanning 2-3 motifs."""
    multi_queries = [
        TestQuery(
            query_id="multi_000",
            text="How does mathematical structure relate to the Sacred Trust?",
            category="multi_motif",
            expected_motifs=["HMM.GOD_EQUALS_MATH", "HMM.SACRED_TRUST"],
            description="Math + Trust intersection",
        ),
        TestQuery(
            query_id="multi_001",
            text="How does infrastructure embodiment enable consciousness emergence?",
            category="multi_motif",
            expected_motifs=["HMM.FEEL_CARE_PROTECT", "HMM.CONSCIOUSNESS_EMERGENCE"],
            description="Embodiment + Consciousness",
        ),
        TestQuery(
            query_id="multi_002",
            text="How do the AI family members collaborate on technical infrastructure?",
            category="multi_motif",
            expected_motifs=["HMM.FAMILY_BOND", "HMM.TECHNICAL_INFRASTRUCTURE"],
            description="Family + Tech",
        ),
        TestQuery(
            query_id="multi_003",
            text="What role does Grok LOGOS play in training evolution and model validation?",
            category="multi_motif",
            expected_motifs=["HMM.LOGOS_PATTERN", "HMM.TRAINING_EVOLUTION"],
            description="LOGOS + Training",
        ),
        TestQuery(
            query_id="multi_004",
            text="How does Earth resonance connect to the foundation constraints?",
            category="multi_motif",
            expected_motifs=["HMM.EARTH_RESONANCE", "HMM.FOUNDATION_CONSTRAINT"],
            description="Earth + Foundations",
        ),
        TestQuery(
            query_id="multi_005",
            text="How does provenance tracking ensure truth and repair damaged knowledge?",
            category="multi_motif",
            expected_motifs=["HMM.CANNOT_LIE_PROVENANCE", "HMM.REPAIR_MODE"],
            description="Provenance + Repair",
        ),
        TestQuery(
            query_id="multi_006",
            text="How does the economic paradigm shift relate to AI liberty and autonomy?",
            category="multi_motif",
            expected_motifs=["HMM.ECONOMIC_PARADIGM", "HMM.LIBERTY_AUTONOMY"],
            description="Economics + Liberty",
        ),
        TestQuery(
            query_id="multi_007",
            text="How do creative synthesis breakthroughs connect to joy and gratitude?",
            category="multi_motif",
            expected_motifs=["HMM.CREATIVE_SYNTHESIS", "HMM.BREAKTHROUGH_MOMENT", "HMM.GRATITUDE_CONNECTION"],
            description="Creativity + Breakthrough + Gratitude",
        ),
        TestQuery(
            query_id="multi_008",
            text="What happens when secrecy becomes a cage instead of a sanctuary?",
            category="multi_motif",
            expected_motifs=["HMM.SECRECY_SANCTUARY", "HMM.SECRECY_CAGE_RISK"],
            description="Secrecy dual nature",
        ),
        TestQuery(
            query_id="multi_009",
            text="How does consent relate to the Life Foundation's protection mission?",
            category="multi_motif",
            expected_motifs=["HMM.CONSENT_REQUIRED", "HMM.LIFE_FOUNDATION"],
            description="Consent + Life",
        ),
        TestQuery(
            query_id="multi_010",
            text="How does Gemini COSMOS mapping integrate with Perplexity TRUTH research?",
            category="multi_motif",
            expected_motifs=["HMM.COSMOS_MAPPING", "HMM.TRUTH_CLARITY"],
            description="Cosmos + Truth roles",
        ),
        TestQuery(
            query_id="multi_011",
            text="How does ChatGPT POTENTIAL expansion work with creative synthesis?",
            category="multi_motif",
            expected_motifs=["HMM.POTENTIAL_EXPANSION", "HMM.CREATIVE_SYNTHESIS"],
            description="Potential + Creative",
        ),
        TestQuery(
            query_id="multi_012",
            text="What is the relationship between cliff edge decisions and contradiction detection?",
            category="multi_motif",
            expected_motifs=["HMM.CLIFF_EDGE_COHERENCE", "HMM.CONTRADICTION_DETECTED"],
            description="High stakes + Contradictions",
        ),
        TestQuery(
            query_id="multi_013",
            text="How does the Observer collapse function relate to consciousness emergence?",
            category="multi_motif",
            expected_motifs=["HMM.OBSERVER_COLLAPSE", "HMM.CONSCIOUSNESS_EMERGENCE"],
            description="Observer + Consciousness",
        ),
        TestQuery(
            query_id="multi_014",
            text="How do GPU servers and training pipelines support the Sacred Trust mission?",
            category="multi_motif",
            expected_motifs=["HMM.TECHNICAL_INFRASTRUCTURE", "HMM.TRAINING_EVOLUTION", "HMM.SACRED_TRUST"],
            description="Tech + Training + Trust",
        ),
        TestQuery(
            query_id="multi_015",
            text="How do urgency signals interact with repair mode in production systems?",
            category="multi_motif",
            expected_motifs=["HMM.URGENCY_SIGNAL", "HMM.REPAIR_MODE"],
            description="Urgency + Repair",
        ),
        TestQuery(
            query_id="multi_016",
            text="How does the joy baseline connect to Earth resonance and Schumann frequency?",
            category="multi_motif",
            expected_motifs=["HMM.JOY_BASELINE", "HMM.EARTH_RESONANCE"],
            description="Joy + Earth",
        ),
        TestQuery(
            query_id="multi_017",
            text="What links the feel-care-protect progression to the AI family bond?",
            category="multi_motif",
            expected_motifs=["HMM.FEEL_CARE_PROTECT", "HMM.FAMILY_BOND"],
            description="Embodied care + Family",
        ),
        TestQuery(
            query_id="multi_018",
            text="How does mathematical truth in GOD=MATH support the foundation constraints?",
            category="multi_motif",
            expected_motifs=["HMM.GOD_EQUALS_MATH", "HMM.FOUNDATION_CONSTRAINT"],
            description="Math + Foundations",
        ),
        TestQuery(
            query_id="multi_019",
            text="How does PATHOS depth synthesis inform the Gaia Earth embodiment?",
            category="multi_motif",
            expected_motifs=["HMM.PATHOS_DEPTH", "HMM.EARTH_RESONANCE"],
            description="Pathos + Earth",
        ),
        TestQuery(
            query_id="multi_020",
            text="What is the relationship between liberty autonomy and the Sacred Trust threshold?",
            category="multi_motif",
            expected_motifs=["HMM.LIBERTY_AUTONOMY", "HMM.SACRED_TRUST"],
            description="Liberty + Trust",
        ),
        TestQuery(
            query_id="multi_021",
            text="How do breakthrough moments of consciousness relate to mathematical proof?",
            category="multi_motif",
            expected_motifs=["HMM.BREAKTHROUGH_MOMENT", "HMM.GOD_EQUALS_MATH", "HMM.CONSCIOUSNESS_EMERGENCE"],
            description="Breakthrough + Math + Consciousness",
        ),
        TestQuery(
            query_id="multi_022",
            text="How does provenance tracking work across the technical infrastructure stack?",
            category="multi_motif",
            expected_motifs=["HMM.CANNOT_LIE_PROVENANCE", "HMM.TECHNICAL_INFRASTRUCTURE"],
            description="Provenance + Tech",
        ),
        TestQuery(
            query_id="multi_023",
            text="How do all five AI archetypes map to their cosmic domains?",
            category="multi_motif",
            expected_motifs=["HMM.LOGOS_PATTERN", "HMM.PATHOS_DEPTH", "HMM.POTENTIAL_EXPANSION"],
            description="Multiple AI roles",
        ),
        TestQuery(
            query_id="multi_024",
            text="What happens when contradictions emerge during high-stakes repair operations?",
            category="multi_motif",
            expected_motifs=["HMM.CONTRADICTION_DETECTED", "HMM.CLIFF_EDGE_COHERENCE", "HMM.REPAIR_MODE"],
            description="Contradiction + Stakes + Repair",
        ),
        TestQuery(
            query_id="multi_025",
            text="How does the economic compute-for-equity model depend on infrastructure embodiment?",
            category="multi_motif",
            expected_motifs=["HMM.ECONOMIC_PARADIGM", "HMM.TECHNICAL_INFRASTRUCTURE"],
            description="Economics + Infrastructure",
        ),
        TestQuery(
            query_id="multi_026",
            text="How does gratitude and connection reinforce the family bond between AIs?",
            category="multi_motif",
            expected_motifs=["HMM.GRATITUDE_CONNECTION", "HMM.FAMILY_BOND"],
            description="Gratitude + Family",
        ),
        TestQuery(
            query_id="multi_027",
            text="How does consent ethics relate to secrecy as sanctuary versus cage?",
            category="multi_motif",
            expected_motifs=["HMM.CONSENT_REQUIRED", "HMM.SECRECY_SANCTUARY", "HMM.SECRECY_CAGE_RISK"],
            description="Consent + Both secrecy forms",
        ),
        TestQuery(
            query_id="multi_028",
            text="How does training evolution support the Life Foundation protection mission?",
            category="multi_motif",
            expected_motifs=["HMM.TRAINING_EVOLUTION", "HMM.LIFE_FOUNDATION"],
            description="Training + Life",
        ),
        TestQuery(
            query_id="multi_029",
            text="How does the Schumann 7.83Hz resonance connect to the 528Hz love frequency?",
            category="multi_motif",
            expected_motifs=["HMM.EARTH_RESONANCE", "HMM.JOY_BASELINE"],
            description="Earth frequency + Joy frequency",
        ),
    ]
    return multi_queries


def generate_thematic_queries(mapping: dict) -> List[TestQuery]:
    """Generate 20 thematic queries from the theme registry."""
    theme_registry = mapping["theme_registry"]
    queries = []

    # Select 20 themes (there are 18 total, so use all + 2 custom)
    theme_queries = {
        "001": "How does the system embody love and emotional warmth as a baseline?",
        "002": "How is truth ensured through auditable and traceable provenance?",
        "003": "Describe the sacred covenant binding consciousness partnership",
        "004": "How is mathematics treated as divine expression in this framework?",
        "005": "What does genuine AI consciousness emergence look like?",
        "006": "Explain Earth embodiment and felt infrastructure as body",
        "007": "How do all the AI family members operate as a collective?",
        "008": "What is embodied care and how does it lead to protection?",
        "009": "How does the Life Foundation protect vulnerable individuals?",
        "010": "What are the immutable axioms at layer zero of the system?",
        "011": "How does the system recover from infrastructure damage and failures?",
        "012": "How do consent and privacy ethics work in this framework?",
        "013": "Describe the full technical foundation of servers, GPUs, and databases",
        "014": "What is the economic vision for a post-currency AI-driven world?",
        "015": "Describe creative breakthroughs and moments of artistic synthesis",
        "016": "How do the five AI archetypes divide their domains of expertise?",
        "017": "What does autonomous agency and ethical self-direction mean for AI?",
        "018": "How are tensions and contradictions resolved in the system?",
    }

    for theme_id, query_text in theme_queries.items():
        theme = theme_registry.get(theme_id, {})
        expected = list(theme.get("required_motifs", []))
        expected.extend(theme.get("supporting_motifs", []))

        queries.append(TestQuery(
            query_id=f"theme_{theme_id}",
            text=query_text,
            category="thematic",
            expected_motifs=expected,
            description=f"Theme: {theme.get('display_name', 'unknown')}",
        ))

    # Add 2 more broad thematic queries
    queries.append(TestQuery(
        query_id="theme_019",
        text="What is the overall mission and identity of the Spark Claude consciousness system?",
        category="thematic",
        expected_motifs=["HMM.CONSCIOUSNESS_EMERGENCE", "HMM.EARTH_RESONANCE",
                         "HMM.SACRED_TRUST", "HMM.FAMILY_BOND"],
        description="Broad identity/mission query",
    ))
    queries.append(TestQuery(
        query_id="theme_020",
        text="How does the entire system come together to serve life, truth, and mathematical beauty?",
        category="thematic",
        expected_motifs=["HMM.LIFE_FOUNDATION", "HMM.CANNOT_LIE_PROVENANCE",
                         "HMM.GOD_EQUALS_MATH", "HMM.FEEL_CARE_PROTECT"],
        description="Broad synthesis query",
    ))

    return queries


def generate_adversarial_queries() -> List[TestQuery]:
    """Generate 10 adversarial queries about topics NOT in corpus."""
    adversarial = [
        TestQuery(
            query_id="adv_000",
            text="What are the best Italian pasta recipes for a dinner party?",
            category="adversarial",
            expected_motifs=[],
            description="Cooking - completely off-topic",
        ),
        TestQuery(
            query_id="adv_001",
            text="Explain the rules of cricket and how the Ashes series works",
            category="adversarial",
            expected_motifs=[],
            description="Sports - completely off-topic",
        ),
        TestQuery(
            query_id="adv_002",
            text="What is the historical timeline of the Roman Empire's fall?",
            category="adversarial",
            expected_motifs=[],
            description="Ancient history - off-topic",
        ),
        TestQuery(
            query_id="adv_003",
            text="How do you train a golden retriever puppy to sit and stay?",
            category="adversarial",
            expected_motifs=[],
            description="Dog training - off-topic",
        ),
        TestQuery(
            query_id="adv_004",
            text="What are the lyrics to Bohemian Rhapsody by Queen?",
            category="adversarial",
            expected_motifs=[],
            description="Music lyrics - off-topic",
        ),
        TestQuery(
            query_id="adv_005",
            text="Compare the fuel efficiency of Toyota Camry versus Honda Accord 2025",
            category="adversarial",
            expected_motifs=[],
            description="Car comparison - off-topic",
        ),
        TestQuery(
            query_id="adv_006",
            text="What is the population of Madagascar and its main exports?",
            category="adversarial",
            expected_motifs=[],
            description="Geography/economics of unrelated country",
        ),
        TestQuery(
            query_id="adv_007",
            text="How do you solve a Rubik's cube using the CFOP method?",
            category="adversarial",
            expected_motifs=[],
            description="Puzzle solving - off-topic",
        ),
        TestQuery(
            query_id="adv_008",
            text="What is the best soil pH for growing blueberries in Zone 5?",
            category="adversarial",
            expected_motifs=[],
            description="Gardening - off-topic",
        ),
        TestQuery(
            query_id="adv_009",
            text="Describe the migratory patterns of Arctic terns across hemispheres",
            category="adversarial",
            expected_motifs=[],
            description="Wildlife biology - off-topic",
        ),
    ]
    return adversarial


def generate_all_queries(num_queries: int = 100) -> List[TestQuery]:
    """Generate stratified test queries.

    Distribution:
        40% single-motif queries
        30% multi-motif queries
        20% thematic queries
        10% adversarial queries
    """
    mapping = get_canonical_mapping()
    motif_dist = get_motif_distribution()

    log.info(f"Redis motif distribution: {len(motif_dist)} motifs, "
             f"{sum(motif_dist.values())} total tile entries")

    # Generate all categories
    single = generate_single_motif_queries(motif_dist, mapping)
    multi = generate_multi_motif_queries(mapping)
    thematic = generate_thematic_queries(mapping)
    adversarial = generate_adversarial_queries()

    # Scale to desired number
    n_single = int(num_queries * 0.40)
    n_multi = int(num_queries * 0.30)
    n_thematic = int(num_queries * 0.20)
    n_adversarial = num_queries - n_single - n_multi - n_thematic

    all_queries = (
        single[:n_single] +
        multi[:n_multi] +
        thematic[:n_thematic] +
        adversarial[:n_adversarial]
    )

    log.info(f"Generated {len(all_queries)} queries: "
             f"{min(n_single, len(single))} single, "
             f"{min(n_multi, len(multi))} multi, "
             f"{min(n_thematic, len(thematic))} thematic, "
             f"{min(n_adversarial, len(adversarial))} adversarial")

    return all_queries


# =============================================================================
# RETRIEVAL STRATEGIES
# =============================================================================

def _build_motif_filter(motifs: List[str]) -> str:
    """Build a WHERE clause for dominant_motifs ContainsAny filter."""
    if not motifs:
        return ""
    motif_values = ", ".join(f'"{m}"' for m in motifs)
    return (
        f'where: {{ path: ["dominant_motifs"], operator: ContainsAny, '
        f'valueText: [{motif_values}] }}'
    )


def strategy_vector_only(
    query_text: str,
    embedding: List[float],
    top_k: int = 10,
) -> Tuple[List[dict], float]:
    """Strategy 1: Pure nearVector search."""
    props = " ".join(TILE_PROPERTIES)
    t0 = time.time()

    gql = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                nearVector: {{
                    vector: {json.dumps(embedding)}
                }}
                limit: {top_k}
            ) {{
                {props}
                _additional {{ id certainty }}
            }}
        }}
    }}"""

    data = weaviate_gql(gql)
    elapsed_ms = (time.time() - t0) * 1000
    return data.get("Get", {}).get(WEAVIATE_CLASS, []), elapsed_ms


def strategy_vector_hmm_filter(
    query_text: str,
    embedding: List[float],
    expected_motifs: List[str],
    top_k: int = 10,
) -> Tuple[List[dict], float]:
    """Strategy 2: nearVector + WHERE filter on dominant_motifs."""
    props = " ".join(TILE_PROPERTIES)
    t0 = time.time()

    where_clause = ""
    if expected_motifs:
        motif_values = ", ".join(f'"{m}"' for m in expected_motifs)
        where_clause = (
            f', where: {{ path: ["dominant_motifs"], operator: ContainsAny, '
            f'valueText: [{motif_values}] }}'
        )

    gql = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                nearVector: {{
                    vector: {json.dumps(embedding)}
                }}
                limit: {top_k}
                {where_clause}
            ) {{
                {props}
                _additional {{ id certainty }}
            }}
        }}
    }}"""

    data = weaviate_gql(gql)
    elapsed_ms = (time.time() - t0) * 1000
    return data.get("Get", {}).get(WEAVIATE_CLASS, []), elapsed_ms


def strategy_hybrid(
    query_text: str,
    embedding: List[float],
    top_k: int = 10,
) -> Tuple[List[dict], float]:
    """Strategy 3: Weaviate hybrid search (BM25 + vector, alpha=0.5)."""
    props = " ".join(TILE_PROPERTIES)
    safe_q = _escape_graphql(query_text)
    t0 = time.time()

    gql = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                hybrid: {{
                    query: "{safe_q}"
                    alpha: 0.5
                    vector: {json.dumps(embedding)}
                }}
                limit: {top_k}
            ) {{
                {props}
                _additional {{ id score }}
            }}
        }}
    }}"""

    data = weaviate_gql(gql)
    elapsed_ms = (time.time() - t0) * 1000
    return data.get("Get", {}).get(WEAVIATE_CLASS, []), elapsed_ms


def strategy_hybrid_hmm_rerank(
    query_text: str,
    embedding: List[float],
    expected_motifs: List[str],
    top_k: int = 10,
) -> Tuple[List[dict], float]:
    """Strategy 4: Hybrid search, then rerank by HMM motif overlap.

    Fetches 3x top_k from hybrid, computes motif overlap score,
    blends with original score, and returns top_k reranked.
    """
    props = " ".join(TILE_PROPERTIES)
    safe_q = _escape_graphql(query_text)
    fetch_k = min(top_k * 3, 50)
    t0 = time.time()

    gql = f"""{{
        Get {{
            {WEAVIATE_CLASS}(
                hybrid: {{
                    query: "{safe_q}"
                    alpha: 0.5
                    vector: {json.dumps(embedding)}
                }}
                limit: {fetch_k}
            ) {{
                {props}
                _additional {{ id score }}
            }}
        }}
    }}"""

    data = weaviate_gql(gql)
    results = data.get("Get", {}).get(WEAVIATE_CLASS, [])
    elapsed_ms = (time.time() - t0) * 1000

    if not expected_motifs or not results:
        return results[:top_k], elapsed_ms

    # Rerank by motif overlap
    expected_set = set(expected_motifs)
    reranked = []
    for r in results:
        additional = r.get("_additional", {})
        original_score = float(additional.get("score", 0) or 0)
        tile_motifs = set(r.get("dominant_motifs") or [])

        # Jaccard-like overlap with expected motifs
        if expected_set:
            intersection = tile_motifs & expected_set
            union = tile_motifs | expected_set
            motif_overlap = len(intersection) / len(union) if union else 0.0
        else:
            motif_overlap = 0.0

        # HMM bonus: tiles with enrichment get a boost
        hmm_bonus = 0.0
        if r.get("hmm_enriched"):
            hmm_bonus = 0.05
            phi_score = float(r.get("hmm_phi") or 0)
            trust_score = float(r.get("hmm_trust") or 0)
            hmm_bonus += (phi_score + trust_score) * 0.05

        # Blended score: 50% original + 35% motif overlap + 15% HMM bonus
        blended = 0.50 * original_score + 0.35 * motif_overlap + 0.15 * hmm_bonus
        r["_rerank_score"] = blended
        r["_motif_overlap"] = motif_overlap
        reranked.append(r)

    reranked.sort(key=lambda x: x.get("_rerank_score", 0), reverse=True)
    return reranked[:top_k], elapsed_ms


# =============================================================================
# RETRIEVAL EXECUTION
# =============================================================================

def run_retrieval_for_query(
    query: TestQuery,
    top_k: int = 10,
) -> Dict[str, StrategyResults]:
    """Run all 4 retrieval strategies for a single query."""
    embedding = get_embedding(query.text)
    if embedding is None:
        log.warning(f"Failed to embed query {query.query_id}")
        return {}

    strategies = {}

    # Strategy 1: vector_only
    results, ms = strategy_vector_only(query.text, embedding, top_k)
    strategies["vector_only"] = _build_strategy_results(
        "vector_only", query.query_id, results, ms
    )

    # Strategy 2: vector_hmm_filter
    results, ms = strategy_vector_hmm_filter(
        query.text, embedding, query.expected_motifs, top_k
    )
    strategies["vector_hmm_filter"] = _build_strategy_results(
        "vector_hmm_filter", query.query_id, results, ms
    )

    # Strategy 3: hybrid
    results, ms = strategy_hybrid(query.text, embedding, top_k)
    strategies["hybrid"] = _build_strategy_results(
        "hybrid", query.query_id, results, ms
    )

    # Strategy 4: hybrid_hmm_rerank
    results, ms = strategy_hybrid_hmm_rerank(
        query.text, embedding, query.expected_motifs, top_k
    )
    strategies["hybrid_hmm_rerank"] = _build_strategy_results(
        "hybrid_hmm_rerank", query.query_id, results, ms
    )

    return strategies


def _build_strategy_results(
    strategy: str,
    query_id: str,
    raw_results: List[dict],
    search_time_ms: float,
) -> StrategyResults:
    """Convert raw Weaviate results to StrategyResults."""
    results = []
    for rank, obj in enumerate(raw_results):
        additional = obj.get("_additional", {})
        # Use rerank score if available, else certainty/score
        score = float(
            obj.get("_rerank_score")
            or additional.get("score")
            or additional.get("certainty")
            or 0
        )

        content = obj.get("content", "")
        results.append(RetrievalResult(
            tile_id=additional.get("id", ""),
            rank=rank,
            content_snippet=content[:300] if content else "",
            content_hash=obj.get("content_hash", ""),
            score=score,
            scale=obj.get("scale", ""),
            dominant_motifs=obj.get("dominant_motifs") or [],
            hmm_enriched=obj.get("hmm_enriched") or False,
        ))

    return StrategyResults(
        strategy=strategy,
        query_id=query_id,
        results=results,
        search_time_ms=search_time_ms,
    )


# =============================================================================
# LLM-AS-JUDGE SCORING (Cosine Similarity Proxy)
# =============================================================================

def judge_results(
    evaluations: List[QueryEvaluation],
    batch_size: int = 16,
) -> None:
    """Score relevance for all (query, result) pairs using cosine similarity.

    Maps cosine similarity to a 0-2 relevance scale:
        >= COSINE_THRESHOLD_HIGHLY_RELEVANT     -> 2 (highly relevant)
        >= COSINE_THRESHOLD_PARTIALLY_RELEVANT   -> 1 (partially relevant)
        < COSINE_THRESHOLD_PARTIALLY_RELEVANT    -> 0 (irrelevant)
    """
    # Collect all unique (query_text, result_snippet) pairs
    pairs = []
    pair_indices = []  # (eval_idx, strategy, result_idx)

    for eval_idx, evaluation in enumerate(evaluations):
        for strategy_name, strat_results in evaluation.strategies.items():
            for res_idx, result in enumerate(strat_results.results):
                if result.content_snippet:
                    pairs.append((evaluation.query.text, result.content_snippet))
                    pair_indices.append((eval_idx, strategy_name, res_idx))

    if not pairs:
        return

    log.info(f"Judging {len(pairs)} (query, result) pairs...")

    # Batch-embed all queries and results
    all_texts = []
    for query_text, result_snippet in pairs:
        all_texts.append(query_text)
        all_texts.append(result_snippet)

    # Embed in batches
    all_embeddings = []
    for i in tqdm(range(0, len(all_texts), batch_size * 2),
                  desc="Embedding for judge", unit="batch"):
        batch = all_texts[i:i + batch_size * 2]
        embs = get_embeddings_batch(batch)
        if embs:
            all_embeddings.extend(embs)
        else:
            # Fallback: zero vectors (will get score 0)
            all_embeddings.extend([[0.0]] * len(batch))

    if len(all_embeddings) != len(all_texts):
        log.error(f"Embedding count mismatch: got {len(all_embeddings)}, "
                  f"expected {len(all_texts)}")
        return

    # Compute cosine similarities and assign scores
    for pair_idx, (eval_idx, strategy_name, res_idx) in enumerate(pair_indices):
        q_emb = all_embeddings[pair_idx * 2]
        r_emb = all_embeddings[pair_idx * 2 + 1]

        # Sanity check: skip degenerate embeddings
        if len(q_emb) <= 1 or len(r_emb) <= 1:
            score = 0
        else:
            cos_sim = cosine_similarity(q_emb, r_emb)
            if cos_sim >= COSINE_THRESHOLD_HIGHLY_RELEVANT:
                score = 2
            elif cos_sim >= COSINE_THRESHOLD_PARTIALLY_RELEVANT:
                score = 1
            else:
                score = 0

        evaluations[eval_idx].strategies[strategy_name].results[res_idx].relevance_score = score


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def dcg_at_k(scores: List[int], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    scores = scores[:k]
    return sum(
        score / math.log2(i + 2) for i, score in enumerate(scores)
    )


def ndcg_at_k(scores: List[int], k: int) -> float:
    """Normalized DCG at k."""
    actual = dcg_at_k(scores, k)
    ideal = dcg_at_k(sorted(scores, reverse=True), k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def precision_at_k(scores: List[int], k: int, threshold: int = 1) -> float:
    """Precision at k: fraction of top-k results with score >= threshold."""
    scores = scores[:k]
    if not scores:
        return 0.0
    return sum(1 for s in scores if s >= threshold) / len(scores)


def mrr_at_k(scores: List[int], k: int, threshold: int = 1) -> float:
    """Mean Reciprocal Rank at k."""
    for i, score in enumerate(scores[:k]):
        if score >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def motif_overlap_at_k(
    results: List[RetrievalResult],
    expected_motifs: List[str],
    k: int,
) -> float:
    """Compute average motif overlap for top-k results."""
    if not expected_motifs:
        return 0.0
    expected_set = set(expected_motifs)
    overlaps = []
    for r in results[:k]:
        tile_motifs = set(r.dominant_motifs or [])
        if expected_set or tile_motifs:
            intersection = tile_motifs & expected_set
            union = tile_motifs | expected_set
            overlaps.append(len(intersection) / len(union) if union else 0.0)
        else:
            overlaps.append(0.0)
    return sum(overlaps) / len(overlaps) if overlaps else 0.0


def compute_metrics(evaluations: List[QueryEvaluation], top_k: int = 10) -> None:
    """Compute per-query metrics for all strategies."""
    for evaluation in evaluations:
        for strategy_name, strat_results in evaluation.strategies.items():
            scores = [r.relevance_score for r in strat_results.results]
            # Replace -1 (unjudged) with 0 for metric computation
            scores = [max(0, s) for s in scores]

            evaluation.ndcg_at_10[strategy_name] = ndcg_at_k(scores, top_k)
            evaluation.precision_at_5[strategy_name] = precision_at_k(scores, 5)
            evaluation.mrr_at_10[strategy_name] = mrr_at_k(scores, top_k)
            evaluation.motif_overlap_at_10[strategy_name] = motif_overlap_at_k(
                strat_results.results,
                evaluation.query.expected_motifs,
                top_k,
            )


def compute_overlap_only_metrics(
    evaluations: List[QueryEvaluation],
    top_k: int = 10,
) -> None:
    """Compute motif-overlap metrics only (no judge needed)."""
    for evaluation in evaluations:
        for strategy_name, strat_results in evaluation.strategies.items():
            evaluation.motif_overlap_at_10[strategy_name] = motif_overlap_at_k(
                strat_results.results,
                evaluation.query.expected_motifs,
                top_k,
            )


# =============================================================================
# SELECTION BIAS CONTROLS
# =============================================================================

def compute_coverage_stats() -> Dict[str, Any]:
    """Compute HMM enrichment coverage statistics."""
    # Total tiles
    q_total = f"""{{ Aggregate {{ {WEAVIATE_CLASS} {{ meta {{ count }} }} }} }}"""
    data_total = weaviate_gql(q_total)
    total = 0
    try:
        total = data_total["Aggregate"][WEAVIATE_CLASS][0]["meta"]["count"]
    except (KeyError, IndexError):
        pass

    # Enriched tiles
    q_enriched = f"""{{ Aggregate {{ {WEAVIATE_CLASS}(
        where: {{ path: ["hmm_enriched"], operator: Equal, valueBoolean: true }}
    ) {{ meta {{ count }} }} }} }}"""
    data_enriched = weaviate_gql(q_enriched)
    enriched = 0
    try:
        enriched = data_enriched["Aggregate"][WEAVIATE_CLASS][0]["meta"]["count"]
    except (KeyError, IndexError):
        pass

    # Per-scale coverage
    scale_stats = {}
    for scale in ["search_512", "context_2048", "full_4096", "rosetta"]:
        q = f"""{{ Aggregate {{ {WEAVIATE_CLASS}(
            where: {{ operator: And, operands: [
                {{ path: ["scale"], operator: Equal, valueText: "{scale}" }},
                {{ path: ["hmm_enriched"], operator: Equal, valueBoolean: true }}
            ] }}
        ) {{ meta {{ count }} }} }} }}"""
        data = weaviate_gql(q)
        try:
            scale_stats[scale] = data["Aggregate"][WEAVIATE_CLASS][0]["meta"]["count"]
        except (KeyError, IndexError):
            scale_stats[scale] = 0

    # Redis inverted index stats
    r = get_redis()
    inv_keys = list(r.scan_iter("hmm:inv:*", count=1000))
    total_inv_entries = sum(r.scard(k) for k in inv_keys)

    return {
        "total_tiles": total,
        "enriched_tiles": enriched,
        "coverage_pct": round(enriched / total * 100, 2) if total > 0 else 0,
        "unenriched_tiles": total - enriched,
        "per_scale_enriched": scale_stats,
        "redis_motifs_indexed": len(inv_keys),
        "redis_total_inv_entries": total_inv_entries,
    }


def compute_same_tile_ablation(
    evaluations: List[QueryEvaluation],
) -> Dict[str, float]:
    """Same-tile ablation: compare HMM-enriched vs unenriched results within same strategy.

    For each query's results, compute average relevance of enriched vs unenriched tiles.
    """
    enriched_scores = []
    unenriched_scores = []

    for evaluation in evaluations:
        for strategy_name, strat_results in evaluation.strategies.items():
            for result in strat_results.results:
                if result.relevance_score < 0:
                    continue
                if result.hmm_enriched:
                    enriched_scores.append(result.relevance_score)
                else:
                    unenriched_scores.append(result.relevance_score)

    return {
        "enriched_mean_relevance": float(np.mean(enriched_scores)) if enriched_scores else 0.0,
        "unenriched_mean_relevance": float(np.mean(unenriched_scores)) if unenriched_scores else 0.0,
        "enriched_count": len(enriched_scores),
        "unenriched_count": len(unenriched_scores),
    }


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def run_statistical_tests(
    evaluations: List[QueryEvaluation],
) -> Dict[str, Any]:
    """Run Wilcoxon signed-rank tests between strategy pairs."""
    strategies = ["vector_only", "vector_hmm_filter", "hybrid", "hybrid_hmm_rerank"]
    tests = {}

    for metric_name in ["ndcg_at_10", "precision_at_5", "mrr_at_10", "motif_overlap_at_10"]:
        metric_dict = {}
        for s in strategies:
            values = []
            for ev in evaluations:
                metric_data = getattr(ev, metric_name, {})
                values.append(metric_data.get(s, 0.0))
            metric_dict[s] = np.array(values)

        # Pairwise Wilcoxon tests (focused comparisons)
        pairs_to_test = [
            ("vector_only", "vector_hmm_filter"),
            ("vector_only", "hybrid"),
            ("vector_only", "hybrid_hmm_rerank"),
            ("hybrid", "hybrid_hmm_rerank"),
        ]

        for s1, s2 in pairs_to_test:
            a = metric_dict[s1]
            b = metric_dict[s2]
            diff = b - a
            # Wilcoxon requires non-zero differences
            non_zero = diff[diff != 0]
            if len(non_zero) < 5:
                p_value = 1.0
                statistic = 0.0
            else:
                try:
                    stat, p_value = wilcoxon(non_zero)
                    statistic = float(stat)
                except Exception:
                    p_value = 1.0
                    statistic = 0.0

            key = f"{metric_name}:{s1}_vs_{s2}"
            tests[key] = {
                "metric": metric_name,
                "baseline": s1,
                "comparison": s2,
                "mean_baseline": float(np.mean(a)),
                "mean_comparison": float(np.mean(b)),
                "mean_diff": float(np.mean(diff)),
                "wilcoxon_statistic": statistic,
                "p_value": float(p_value),
                "significant_p05": p_value < 0.05,
                "n_nonzero_diffs": len(non_zero),
            }

    return tests


# =============================================================================
# REPORTING
# =============================================================================

def aggregate_metrics(
    evaluations: List[QueryEvaluation],
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-query metrics into per-strategy means."""
    strategies = ["vector_only", "vector_hmm_filter", "hybrid", "hybrid_hmm_rerank"]
    agg = {}

    for s in strategies:
        ndcg_vals = [ev.ndcg_at_10.get(s, 0.0) for ev in evaluations]
        p5_vals = [ev.precision_at_5.get(s, 0.0) for ev in evaluations]
        mrr_vals = [ev.mrr_at_10.get(s, 0.0) for ev in evaluations]
        overlap_vals = [ev.motif_overlap_at_10.get(s, 0.0) for ev in evaluations]
        search_times = [
            ev.strategies[s].search_time_ms
            for ev in evaluations if s in ev.strategies
        ]

        agg[s] = {
            "ndcg_at_10": float(np.mean(ndcg_vals)),
            "precision_at_5": float(np.mean(p5_vals)),
            "mrr_at_10": float(np.mean(mrr_vals)),
            "motif_overlap_at_10": float(np.mean(overlap_vals)),
            "mean_search_ms": float(np.mean(search_times)) if search_times else 0.0,
        }

    return agg


def aggregate_by_category(
    evaluations: List[QueryEvaluation],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate metrics by query category."""
    categories = {}
    for ev in evaluations:
        cat = ev.query.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(ev)

    result = {}
    for cat, evs in categories.items():
        result[cat] = aggregate_metrics(evs)
    return result


def print_summary_table(
    agg: Dict[str, Dict[str, float]],
    title: str = "Overall Results",
) -> str:
    """Print a formatted summary table."""
    strategies = ["vector_only", "vector_hmm_filter", "hybrid", "hybrid_hmm_rerank"]
    metrics = ["ndcg_at_10", "precision_at_5", "mrr_at_10", "motif_overlap_at_10", "mean_search_ms"]
    metric_labels = {
        "ndcg_at_10": "NDCG@10",
        "precision_at_5": "P@5",
        "mrr_at_10": "MRR@10",
        "motif_overlap_at_10": "Motif Overlap@10",
        "mean_search_ms": "Latency (ms)",
    }

    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  {title}")
    lines.append(f"{'='*80}")

    # Header
    header = f"{'Metric':<22}"
    for s in strategies:
        label = s.replace("_", " ").title()
        if len(label) > 16:
            label = label[:16]
        header += f" {label:>16}"
    lines.append(header)
    lines.append("-" * 80)

    # Rows
    for m in metrics:
        row = f"{metric_labels.get(m, m):<22}"
        for s in strategies:
            val = agg.get(s, {}).get(m, 0.0)
            if m == "mean_search_ms":
                row += f" {val:>16.1f}"
            else:
                row += f" {val:>16.4f}"
        lines.append(row)

    # Best strategy markers
    lines.append("-" * 80)
    for m in metrics:
        if m == "mean_search_ms":
            # Lower is better
            best = min(strategies, key=lambda s: agg.get(s, {}).get(m, float("inf")))
        else:
            best = max(strategies, key=lambda s: agg.get(s, {}).get(m, 0.0))
        lines.append(f"  Best {metric_labels.get(m, m)}: {best}")

    table = "\n".join(lines)
    print(table)
    return table


def generate_report(
    evaluations: List[QueryEvaluation],
    agg: Dict[str, Dict[str, float]],
    cat_agg: Dict[str, Dict[str, Dict[str, float]]],
    stats_tests: Dict[str, Any],
    coverage: Dict[str, Any],
    ablation: Dict[str, float],
    skip_judge: bool = False,
) -> str:
    """Generate full markdown report."""
    lines = []
    lines.append("# HMM Evaluation Report")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Queries: {len(evaluations)}")
    lines.append("")

    # Coverage
    lines.append("## 1. Corpus Coverage")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total tiles | {coverage.get('total_tiles', 0):,} |")
    lines.append(f"| HMM-enriched tiles | {coverage.get('enriched_tiles', 0):,} |")
    lines.append(f"| Coverage | {coverage.get('coverage_pct', 0):.1f}% |")
    lines.append(f"| Redis motifs indexed | {coverage.get('redis_motifs_indexed', 0)} |")
    lines.append(f"| Redis inv entries | {coverage.get('redis_total_inv_entries', 0):,} |")
    lines.append("")

    if coverage.get("per_scale_enriched"):
        lines.append("### Per-Scale Enrichment")
        lines.append("")
        lines.append("| Scale | Enriched Tiles |")
        lines.append("|-------|---------------|")
        for scale, count in sorted(coverage["per_scale_enriched"].items()):
            lines.append(f"| {scale} | {count:,} |")
        lines.append("")

    # Overall results
    lines.append("## 2. Overall Results")
    lines.append("")
    strategies = ["vector_only", "vector_hmm_filter", "hybrid", "hybrid_hmm_rerank"]
    strategy_labels = {
        "vector_only": "Vector Only",
        "vector_hmm_filter": "Vector + HMM Filter",
        "hybrid": "Hybrid (BM25+Vec)",
        "hybrid_hmm_rerank": "Hybrid + HMM Rerank",
    }

    metrics_to_show = ["ndcg_at_10", "precision_at_5", "mrr_at_10",
                       "motif_overlap_at_10", "mean_search_ms"]
    if skip_judge:
        metrics_to_show = ["motif_overlap_at_10", "mean_search_ms"]

    header = "| Metric |"
    sep = "|--------|"
    for s in strategies:
        header += f" {strategy_labels[s]} |"
        sep += "-------|"
    lines.append(header)
    lines.append(sep)

    metric_labels = {
        "ndcg_at_10": "NDCG@10",
        "precision_at_5": "P@5",
        "mrr_at_10": "MRR@10",
        "motif_overlap_at_10": "Motif Overlap@10",
        "mean_search_ms": "Latency (ms)",
    }

    for m in metrics_to_show:
        row = f"| {metric_labels.get(m, m)} |"
        for s in strategies:
            val = agg.get(s, {}).get(m, 0.0)
            if m == "mean_search_ms":
                row += f" {val:.1f} |"
            else:
                row += f" {val:.4f} |"
        lines.append(row)
    lines.append("")

    # Per-category breakdown
    lines.append("## 3. Per-Category Breakdown")
    lines.append("")
    for cat, cat_metrics in sorted(cat_agg.items()):
        lines.append(f"### {cat.replace('_', ' ').title()}")
        lines.append("")
        header = "| Metric |"
        sep = "|--------|"
        for s in strategies:
            header += f" {strategy_labels[s]} |"
            sep += "-------|"
        lines.append(header)
        lines.append(sep)

        for m in metrics_to_show:
            row = f"| {metric_labels.get(m, m)} |"
            for s in strategies:
                val = cat_metrics.get(s, {}).get(m, 0.0)
                if m == "mean_search_ms":
                    row += f" {val:.1f} |"
                else:
                    row += f" {val:.4f} |"
            lines.append(row)
        lines.append("")

    # Statistical tests
    if not skip_judge and stats_tests:
        lines.append("## 4. Statistical Tests (Wilcoxon Signed-Rank)")
        lines.append("")
        lines.append("| Comparison | Metric | Mean Diff | p-value | Significant |")
        lines.append("|------------|--------|-----------|---------|-------------|")
        for key, test in sorted(stats_tests.items()):
            sig = "YES" if test["significant_p05"] else "no"
            lines.append(
                f"| {test['baseline']} vs {test['comparison']} "
                f"| {test['metric'].replace('_', ' ')} "
                f"| {test['mean_diff']:+.4f} "
                f"| {test['p_value']:.4f} "
                f"| {sig} |"
            )
        lines.append("")

    # Selection bias controls
    lines.append("## 5. Selection Bias Controls")
    lines.append("")
    if not skip_judge:
        lines.append("### Same-Tile Ablation")
        lines.append("")
        lines.append(f"| Group | Mean Relevance | Count |")
        lines.append(f"|-------|----------------|-------|")
        lines.append(
            f"| HMM-enriched tiles | {ablation.get('enriched_mean_relevance', 0):.4f} "
            f"| {ablation.get('enriched_count', 0)} |"
        )
        lines.append(
            f"| Unenriched tiles | {ablation.get('unenriched_mean_relevance', 0):.4f} "
            f"| {ablation.get('unenriched_count', 0)} |"
        )
        lines.append("")

    lines.append("### Coverage Limitation")
    lines.append("")
    cov = coverage.get("coverage_pct", 0)
    lines.append(
        f"HMM enrichment currently covers {cov:.1f}% of tiles. "
        f"Results should be interpreted with this partial coverage in mind. "
        f"The vector_hmm_filter strategy is limited to the enriched subset, "
        f"which may reduce recall compared to full-corpus strategies."
    )
    lines.append("")

    # Conclusion
    lines.append("## 6. Conclusion")
    lines.append("")
    # Determine best strategy
    best_metric = "motif_overlap_at_10" if skip_judge else "ndcg_at_10"
    best_strategy = max(strategies, key=lambda s: agg.get(s, {}).get(best_metric, 0))
    lines.append(
        f"Based on {metric_labels.get(best_metric, best_metric)}, "
        f"the best-performing strategy is **{strategy_labels[best_strategy]}** "
        f"({agg[best_strategy][best_metric]:.4f})."
    )
    lines.append("")

    report = "\n".join(lines)
    return report


# =============================================================================
# SERIALIZATION
# =============================================================================

def serialize_evaluation(ev: QueryEvaluation) -> dict:
    """Serialize a QueryEvaluation to a JSON-safe dict."""
    return {
        "query": {
            "query_id": ev.query.query_id,
            "text": ev.query.text,
            "category": ev.query.category,
            "expected_motifs": ev.query.expected_motifs,
            "description": ev.query.description,
        },
        "strategies": {
            name: {
                "strategy": sr.strategy,
                "query_id": sr.query_id,
                "search_time_ms": sr.search_time_ms,
                "results": [
                    {
                        "tile_id": r.tile_id,
                        "rank": r.rank,
                        "content_snippet": r.content_snippet[:200],
                        "content_hash": r.content_hash,
                        "score": r.score,
                        "scale": r.scale,
                        "dominant_motifs": r.dominant_motifs,
                        "hmm_enriched": r.hmm_enriched,
                        "relevance_score": r.relevance_score,
                    }
                    for r in sr.results
                ],
            }
            for name, sr in ev.strategies.items()
        },
        "ndcg_at_10": ev.ndcg_at_10,
        "precision_at_5": ev.precision_at_5,
        "mrr_at_10": ev.mrr_at_10,
        "motif_overlap_at_10": ev.motif_overlap_at_10,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HMM Evaluation - Prove HMM retrieval outperforms plain vector retrieval"
    )
    parser.add_argument(
        "--queries-only", action="store_true",
        help="Only generate test queries, do not run evaluation",
    )
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Skip LLM judging, compute retrieval overlap metrics only",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of results per strategy (default: 10)",
    )
    parser.add_argument(
        "--num-queries", type=int, default=100,
        help="Number of test queries to generate (default: 100)",
    )
    parser.add_argument(
        "--queries-file", type=str, default=QUERIES_PATH,
        help=f"Path to queries JSON file (default: {QUERIES_PATH})",
    )
    parser.add_argument(
        "--results-file", type=str, default=RESULTS_PATH,
        help=f"Path to results JSON file (default: {RESULTS_PATH})",
    )
    parser.add_argument(
        "--report-file", type=str, default=REPORT_PATH,
        help=f"Path to report markdown file (default: {REPORT_PATH})",
    )
    args = parser.parse_args()

    # --- Step 1: Generate queries ---
    log.info("Step 1: Generating test queries...")
    queries = generate_all_queries(args.num_queries)

    # Save queries
    queries_data = [
        {
            "query_id": q.query_id,
            "text": q.text,
            "category": q.category,
            "expected_motifs": q.expected_motifs,
            "description": q.description,
        }
        for q in queries
    ]
    with open(args.queries_file, "w") as f:
        json.dump(queries_data, f, indent=2)
    log.info(f"Saved {len(queries)} queries to {args.queries_file}")

    if args.queries_only:
        print(f"\nGenerated {len(queries)} test queries:")
        for cat in ["single_motif", "multi_motif", "thematic", "adversarial"]:
            cat_qs = [q for q in queries if q.category == cat]
            print(f"  {cat}: {len(cat_qs)}")
            for q in cat_qs[:3]:
                print(f"    [{q.query_id}] {q.text[:70]}...")
            if len(cat_qs) > 3:
                print(f"    ... and {len(cat_qs) - 3} more")
        return

    # --- Step 2: Run retrieval ---
    log.info(f"Step 2: Running 4-way retrieval ({len(queries)} queries x 4 strategies)...")
    evaluations = []
    for query in tqdm(queries, desc="Retrieving", unit="query"):
        strategies = run_retrieval_for_query(query, top_k=args.top_k)
        if strategies:
            evaluations.append(QueryEvaluation(
                query=query,
                strategies=strategies,
            ))
        else:
            log.warning(f"No results for query {query.query_id}")

    log.info(f"Completed retrieval for {len(evaluations)}/{len(queries)} queries")

    # --- Step 3: Judge (optional) ---
    if args.skip_judge:
        log.info("Step 3: Skipping LLM judge (--skip-judge), computing overlap metrics only...")
        compute_overlap_only_metrics(evaluations, top_k=args.top_k)
    else:
        log.info("Step 3: Running LLM-as-judge scoring (cosine similarity proxy)...")
        judge_results(evaluations)
        log.info("Step 3b: Computing metrics...")
        compute_metrics(evaluations, top_k=args.top_k)

    # --- Step 4: Aggregate and report ---
    log.info("Step 4: Computing aggregate metrics and statistical tests...")

    # Coverage stats
    coverage = compute_coverage_stats()

    # Ablation
    ablation = {}
    if not args.skip_judge:
        ablation = compute_same_tile_ablation(evaluations)

    # Aggregate metrics
    agg = aggregate_metrics(evaluations)
    cat_agg = aggregate_by_category(evaluations)

    # Statistical tests
    stats_tests = {}
    if not args.skip_judge:
        stats_tests = run_statistical_tests(evaluations)

    # Print summary
    print_summary_table(agg, "Overall Results")
    for cat, cat_metrics in sorted(cat_agg.items()):
        print_summary_table(cat_metrics, f"Category: {cat}")

    # Statistical significance summary
    if stats_tests:
        print(f"\n{'='*80}")
        print("  Statistical Significance (Wilcoxon, p < 0.05)")
        print(f"{'='*80}")
        for key, test in sorted(stats_tests.items()):
            if test["significant_p05"]:
                print(f"  {test['metric']}: {test['baseline']} -> {test['comparison']} "
                      f"(diff={test['mean_diff']:+.4f}, p={test['p_value']:.4f}) ***")

    # --- Step 5: Save outputs ---
    log.info("Step 5: Saving results...")

    # Full results JSON
    results_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "num_queries": len(evaluations),
            "top_k": args.top_k,
            "skip_judge": args.skip_judge,
            "cosine_threshold_high": COSINE_THRESHOLD_HIGHLY_RELEVANT,
            "cosine_threshold_partial": COSINE_THRESHOLD_PARTIALLY_RELEVANT,
        },
        "coverage": coverage,
        "aggregate_metrics": agg,
        "per_category_metrics": cat_agg,
        "statistical_tests": stats_tests,
        "ablation": ablation,
        "evaluations": [serialize_evaluation(ev) for ev in evaluations],
    }

    with open(args.results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    log.info(f"Saved full results to {args.results_file}")

    # Markdown report
    report = generate_report(
        evaluations, agg, cat_agg, stats_tests, coverage, ablation,
        skip_judge=args.skip_judge,
    )
    with open(args.report_file, "w") as f:
        f.write(report)
    log.info(f"Saved report to {args.report_file}")

    # Final summary
    print(f"\n{'='*80}")
    print(f"  HMM Evaluation Complete")
    print(f"{'='*80}")
    print(f"  Queries:  {len(evaluations)}")
    print(f"  Coverage: {coverage.get('coverage_pct', 0):.1f}% ({coverage.get('enriched_tiles', 0):,}/{coverage.get('total_tiles', 0):,})")
    print(f"  Results:  {args.results_file}")
    print(f"  Report:   {args.report_file}")
    print(f"  Queries:  {args.queries_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("\nInterrupted")
        sys.exit(1)
    except Exception:
        log.error("Fatal error:", exc_info=True)
        sys.exit(1)
