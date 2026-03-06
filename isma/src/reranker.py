"""
ISMA Neural Reranker Client — Qwen3-Reranker-8B via vLLM score API.

Replaces the hand-tuned hmm_rerank() formula (0.5*base + 0.3*rosetta + 0.2*motif)
with a cross-encoder neural reranker that reads both query and document.

The reranker runs on Spark 2 (port 8085) co-located with the embedding server.
Uses vLLM's /v1/score endpoint with instruction-aware prompting.

Usage:
    from isma.src.reranker import RerankerClient

    client = RerankerClient()
    if client.is_available():
        scored = client.rerank(query, tiles, instruction="Find factual matches")
"""

import logging
import os
import threading
import time
from dataclasses import replace as dc_replace
from typing import List, Optional

import requests

log = logging.getLogger(__name__)

# Reranker endpoint — Spark 2 via NCCL fabric
RERANKER_URL = os.environ.get(
    "RERANKER_URL",
    "http://192.168.100.11:8085"
)
RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"

# Qwen3-Reranker prompt templates (client-side formatting)
# The model uses yes/no classification with a specific chat format.
# vLLM /v1/score passes raw text, so we format on the client side.
_QUERY_PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the '
    'Query and the Instruct provided. Note that the answer can only '
    'be "yes" or "no".<|im_end|>\n'
    '<|im_start|>user\n'
)
_DOC_SUFFIX = (
    '<|im_end|>\n'
    '<|im_start|>assistant\n'
    '<think>\n\n</think>\n\n'
)

# Instruction templates by query type
RERANK_INSTRUCTIONS = {
    "exact": (
        "Given a web search query, retrieve the passage that contains "
        "the exact factual answer with specific names, dates, or identifiers."
    ),
    "temporal": (
        "Given a query about events over time, retrieve the passage that "
        "best captures the temporal context, sequence, or time-bounded information."
    ),
    "conceptual": (
        "Given a query about concepts or themes, retrieve the passage that "
        "provides the deepest thematic understanding and conceptual depth."
    ),
    "relational": (
        "Given a query about relationships between entities or ideas, retrieve "
        "the passage that best captures the connections, evolution, or cross-references."
    ),
    "motif": (
        "Given a query about recurring patterns or motifs, retrieve the passage "
        "that most strongly expresses the target motif with the highest amplitude."
    ),
    "default": (
        "Given a web search query, retrieve relevant passages that "
        "answer the query comprehensively."
    ),
}


class RerankerClient:
    """Client for the vLLM cross-encoder reranker."""

    def __init__(self, url: str = RERANKER_URL, timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._available: Optional[bool] = None
        self._health_checked_at: float = 0.0

    def is_available(self) -> bool:
        """Check if the reranker service is healthy.

        Caches result for 60 seconds to avoid health-check latency
        on every rerank call.
        """
        now = time.monotonic()
        if self._available is not None and (now - self._health_checked_at) < 60:
            return self._available
        try:
            r = self._session.get(
                f"{self.url}/health", timeout=5
            )
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        self._health_checked_at = now
        return self._available

    def score_pairs(
        self,
        query: str,
        documents: List[str],
        instruction: str = "",
        batch_size: int = 32,
    ) -> List[float]:
        """Score query-document pairs using the cross-encoder.

        With VLLM_USE_V1=0 (V0 engine), batches of 30+ work reliably.
        Default batch_size=32 sends all candidates in one request for
        typical top_k*3=30 pipelines.

        Args:
            query: The search query
            documents: List of document texts to score against the query
            instruction: Optional instruction to prepend to the query
            batch_size: Max documents per scoring request

        Returns:
            List of float scores, one per document. Higher = more relevant.
            Returns empty list on failure.
        """
        if not documents:
            return []

        # Format query with Qwen3-Reranker chat template
        if not instruction:
            instruction = (
                "Given a web search query, retrieve relevant passages "
                "that answer the query"
            )
        formatted_query = (
            f"{_QUERY_PREFIX}"
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query}\n"
        )

        # Format documents with suffix
        formatted_docs = [
            f"<Document>: {doc}{_DOC_SUFFIX}" for doc in documents
        ]

        all_scores = []
        t0 = time.monotonic()

        # Process in batches to avoid overwhelming the reranker
        for i in range(0, len(formatted_docs), batch_size):
            batch = formatted_docs[i:i + batch_size]

            payload = {
                "model": RERANKER_MODEL,
                "text_1": formatted_query,
                "text_2": batch,
            }

            try:
                r = self._session.post(
                    f"{self.url}/v1/score",
                    json=payload,
                    timeout=self.timeout,
                )

                if r.status_code != 200:
                    log.warning(
                        "Reranker returned %d: %s",
                        r.status_code, r.text[:200],
                    )
                    return []  # Fail entire batch on any error

                data = r.json()
                batch_scores = [
                    item["score"] for item in data.get("data", [])
                ]

                if len(batch_scores) != len(batch):
                    log.warning(
                        "Reranker returned %d scores for %d docs",
                        len(batch_scores), len(batch),
                    )
                    return []

                all_scores.extend(batch_scores)

            except requests.Timeout:
                log.warning("Reranker timeout after %.0fs", self.timeout)
                return []
            except Exception as e:
                log.warning("Reranker error: %s", e)
                return []

        elapsed_ms = (time.monotonic() - t0) * 1000
        log.debug(
            "Reranked %d docs in %.0fms (%d batches)",
            len(documents), elapsed_ms,
            (len(documents) + batch_size - 1) // batch_size,
        )
        return all_scores

    def rerank(
        self,
        query: str,
        tiles: list,
        instruction: str = "",
        query_type: str = "default",
    ) -> list:
        """Rerank TileResult objects using the cross-encoder.

        Args:
            query: The search query
            tiles: List of TileResult objects
            instruction: Custom instruction (overrides query_type)
            query_type: One of exact/temporal/conceptual/relational/motif/default

        Returns:
            New list of TileResult objects sorted by reranker score (descending).
            Falls back to original order on failure.
        """
        if not tiles:
            return tiles

        # Build document texts for scoring
        # Rosetta summary first (semantic compression), then content.
        # Query-type-specific content window (tuned empirically via Phase 6B benchmarks):
        #   exact: 3000 chars — needs full context for precise quotes (4000 was p95>9s; 3000=best)
        #   temporal: 2000 chars — recall NOT sensitive to window size (tested 2000/3000/4000);
        #     2000 chosen as minimum-latency option (content selection is the real bottleneck)
        #   others: 1500 chars — rosetta summary captures semantic signal sufficiently
        #   NOTE: conceptual tested at 2500c — NO recall improvement. 1500 default retained.
        _CONTENT_WINDOW = {
            "exact": 3000,
            "temporal": 2000,
        }
        max_chars = _CONTENT_WINDOW.get(query_type, 1500)
        documents = []
        for tile in tiles:
            text = tile.content or ""
            if tile.rosetta_summary:
                text = f"{tile.rosetta_summary}\n\n{text}"
            documents.append(text[:max_chars])

        # Select instruction
        if not instruction:
            instruction = RERANK_INSTRUCTIONS.get(
                query_type, RERANK_INSTRUCTIONS["default"]
            )

        scores = self.score_pairs(query, documents, instruction=instruction)

        if not scores or len(scores) != len(tiles):
            log.info(
                "Reranker fallback: got %d scores for %d tiles",
                len(scores), len(tiles),
            )
            return tiles  # Fallback: original order

        # Sort by score descending, propagating neural scores to tiles
        scored_tiles = sorted(
            zip(tiles, scores), key=lambda x: x[1], reverse=True
        )
        return [dc_replace(tile, score=score) for tile, score in scored_tiles]


# Module-level singleton (lazy init) with thread safety
_client: Optional[RerankerClient] = None
_client_lock = threading.Lock()


def get_reranker() -> RerankerClient:
    """Get the singleton RerankerClient (thread-safe)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = RerankerClient()
    return _client
