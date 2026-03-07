"""
Create ISMA_Quantum_v2 Weaviate class with named vectors and BM25F.

The v2 class represents canonical memory objects — one per content_hash
(source document), eliminating the multi-scale tile fragmentation.

Named vectors:
  - raw: from representative search_512 tile embedding (4096d)
  - rosetta: from rosetta_summary embedding (4096d, when available)

BM25F field weighting:
  - rosetta_summary: 3x boost (semantic essence)
  - motif_annotations: 2x boost (motif descriptions as natural language)
  - content: 1x boost (representative 512-token chunk)

Usage:
    python3 -m isma.scripts.create_v2_schema [--delete-existing]
"""
import os

import argparse
import json
import requests
import sys

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://10.0.0.163:8088")
WEAVIATE_GQL = f"{WEAVIATE_URL}/v1/graphql"
WEAVIATE_REST = f"{WEAVIATE_URL}/v1"
CLASS_NAME = "ISMA_Quantum_v2"
VECTOR_DIM = 4096


def create_v2_schema(delete_existing: bool = False):
    """Create the ISMA_Quantum_v2 Weaviate class."""

    # Check if class already exists
    r = requests.get(f"{WEAVIATE_URL}/v1/schema/{CLASS_NAME}")
    if r.status_code == 200:
        if delete_existing:
            print(f"Deleting existing {CLASS_NAME}...")
            requests.delete(f"{WEAVIATE_URL}/v1/schema/{CLASS_NAME}")
        else:
            print(f"{CLASS_NAME} already exists. Use --delete-existing to recreate.")
            return False

    schema = {
        "class": CLASS_NAME,
        "description": (
            "Canonical memory objects — one per content_hash (source document). "
            "Named vectors for multi-representation search. BM25F weighted."
        ),
        "vectorConfig": {
            "raw": {
                "vectorizer": {
                    "none": {}
                },
                "vectorIndexType": "hnsw",
                "vectorIndexConfig": {
                    "distance": "cosine",
                    "ef": -1,
                    "efConstruction": 128,
                    "maxConnections": 32,
                },
            },
            "rosetta": {
                "vectorizer": {
                    "none": {}
                },
                "vectorIndexType": "hnsw",
                "vectorIndexConfig": {
                    "distance": "cosine",
                    "ef": -1,
                    "efConstruction": 128,
                    "maxConnections": 32,
                },
            },
        },
        "invertedIndexConfig": {
            "bm25": {
                "b": 0.75,
                "k1": 1.2,
            },
            "indexTimestamps": True,
        },
        "properties": [
            # Core content fields (BM25F weighted)
            {
                "name": "content",
                "dataType": ["text"],
                "description": "Representative 512-token content chunk (1x BM25 boost)",
                "indexFilterable": True,
                "indexSearchable": True,
                "tokenization": "word",
                "moduleConfig": {
                    "text2vec-none": {
                        "vectorizePropertyName": False,
                    }
                },
            },
            {
                "name": "rosetta_summary",
                "dataType": ["text"],
                "description": "HMM rosetta summary — semantic essence (3x BM25 boost)",
                "indexFilterable": True,
                "indexSearchable": True,
                "tokenization": "word",
            },
            {
                "name": "motif_annotations",
                "dataType": ["text"],
                "description": "Motif definitions as searchable text (2x BM25 boost)",
                "indexFilterable": True,
                "indexSearchable": True,
                "tokenization": "word",
            },
            # Identity and metadata
            {
                "name": "content_hash",
                "dataType": ["text"],
                "description": "Unique identifier for the source document",
                "indexFilterable": True,
                "indexSearchable": False,
                "tokenization": "field",
            },
            {
                "name": "platform",
                "dataType": ["text"],
                "description": "Source platform (claude, grok, gemini, chatgpt, etc)",
                "indexFilterable": True,
                "indexSearchable": False,
                "tokenization": "field",
            },
            {
                "name": "source_type",
                "dataType": ["text"],
                "description": "conversation or corpus",
                "indexFilterable": True,
                "indexSearchable": False,
                "tokenization": "field",
            },
            {
                "name": "source_file",
                "dataType": ["text"],
                "description": "Original source file path",
                "indexFilterable": True,
                "indexSearchable": False,
                "tokenization": "field",
            },
            {
                "name": "session_id",
                "dataType": ["text"],
                "description": "Conversation session identifier",
                "indexFilterable": True,
                "indexSearchable": False,
                "tokenization": "field",
            },
            {
                "name": "document_id",
                "dataType": ["text"],
                "description": "Document identifier",
                "indexFilterable": True,
                "indexSearchable": False,
                "tokenization": "field",
            },
            # HMM enrichment data
            {
                "name": "hmm_enriched",
                "dataType": ["boolean"],
                "description": "Whether this content has been HMM enriched",
                "indexFilterable": True,
            },
            {
                "name": "dominant_motifs",
                "dataType": ["text[]"],
                "description": "List of dominant motif IDs",
                "indexFilterable": True,
                "indexSearchable": False,
                "tokenization": "field",
            },
            {
                "name": "motif_data_json",
                "dataType": ["text"],
                "description": "Full motif data as JSON string",
                "indexFilterable": False,
                "indexSearchable": False,
            },
            {
                "name": "hmm_phi",
                "dataType": ["number"],
                "description": "HMM phi coherence score",
                "indexFilterable": True,
            },
            {
                "name": "hmm_trust",
                "dataType": ["number"],
                "description": "HMM trust score",
                "indexFilterable": True,
            },
            # Tile count and references
            {
                "name": "tile_count_512",
                "dataType": ["int"],
                "description": "Number of search_512 tiles for this content",
                "indexFilterable": True,
            },
            {
                "name": "tile_count_2048",
                "dataType": ["int"],
                "description": "Number of context_2048 tiles for this content",
                "indexFilterable": True,
            },
            {
                "name": "tile_count_4096",
                "dataType": ["int"],
                "description": "Number of full_4096 tiles for this content",
                "indexFilterable": True,
            },
            {
                "name": "total_tokens",
                "dataType": ["int"],
                "description": "Total token count across all tiles",
                "indexFilterable": True,
            },
            # V1 tile UUIDs for detail expansion
            {
                "name": "tile_ids_512",
                "dataType": ["text[]"],
                "description": "V1 search_512 tile UUIDs for passage retrieval",
                "indexFilterable": False,
                "indexSearchable": False,
            },
            {
                "name": "tile_ids_2048",
                "dataType": ["text[]"],
                "description": "V1 context_2048 tile UUIDs for context expansion",
                "indexFilterable": False,
                "indexSearchable": False,
            },
            {
                "name": "tile_ids_4096",
                "dataType": ["text[]"],
                "description": "V1 full_4096 tile UUIDs",
                "indexFilterable": False,
                "indexSearchable": False,
            },
            {
                "name": "rosetta_tile_id",
                "dataType": ["text"],
                "description": "V1 rosetta tile UUID",
                "indexFilterable": False,
                "indexSearchable": False,
                "tokenization": "field",
            },
            # Timestamps
            {
                "name": "loaded_at",
                "dataType": ["text"],
                "description": "When the content was first loaded",
                "indexFilterable": True,
                "indexSearchable": False,
                "tokenization": "field",
            },
            {
                "name": "hmm_enriched_at",
                "dataType": ["text"],
                "description": "When HMM enrichment was last applied",
                "indexFilterable": True,
                "indexSearchable": False,
                "tokenization": "field",
            },
        ],
    }

    r = requests.post(
        f"{WEAVIATE_URL}/v1/schema",
        json=schema,
    )

    if r.status_code == 200:
        print(f"Created {CLASS_NAME} successfully")
        print(f"  Named vectors: raw (4096d), rosetta (4096d)")
        print(f"  BM25F fields: rosetta_summary, motif_annotations, content")
        print(f"  Properties: {len(schema['properties'])}")
        return True
    else:
        print(f"Failed to create {CLASS_NAME}: {r.status_code}")
        print(r.text)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Create {CLASS_NAME} schema")
    parser.add_argument(
        "--delete-existing", action="store_true",
        help="Delete existing class before creating",
    )
    args = parser.parse_args()
    success = create_v2_schema(delete_existing=args.delete_existing)
    sys.exit(0 if success else 1)
