#!/usr/bin/env python3
"""
ISMA Chat Data Migration

Migrates prompt/response/artifact data from Neo4j 7687 (legacy)
into ISMA via Weaviate ISMA_Quantum collection.

Data model in Neo4j 7687:
- Exchange nodes: user_prompt, timestamp, conversation_id
- Response nodes: text, timestamp, sequence
- Relationship: Exchange -[:HAS_RESPONSE]-> Response

Migration approach:
- Query Exchange-Response pairs
- Format as ISMA exchanges
- Tile using φ-tiling (e-based)
- Embed and store in ISMA_Quantum
"""

import os
import sys
import json
import hashlib
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from phi_tiling import phi_tile_text, Tile

# Configuration
NEO4J_LEGACY_URI = "bolt://10.0.0.68:7687"
EMBEDDING_URL = "http://10.0.0.68:8090/embed"
WEAVIATE_URL = "http://10.0.0.68:8088"
BATCH_SIZE = 50  # Exchanges per batch


def get_neo4j_driver():
    """Get Neo4j driver for legacy database."""
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_LEGACY_URI)


def compute_checksum(content: str) -> str:
    """Compute SHA256 checksum of content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def get_embeddings(texts: List[str], batch_size: int = 8, max_retries: int = 3) -> Optional[List[List[float]]]:
    """Get embeddings from embedding server."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        success = False

        for retry in range(max_retries):
            try:
                response = requests.post(
                    EMBEDDING_URL,
                    json={"texts": batch, "batch_size": batch_size},
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()
                all_embeddings.extend(result["embeddings"])
                success = True
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"  Retry {retry + 1}/{max_retries}: {e}")
                    import time
                    time.sleep(2)
                else:
                    print(f"  FAILED batch {i} after {max_retries} retries: {e}")
                    return None

        if not success:
            return None

    return all_embeddings


def store_in_weaviate(tiles: List[Tile], embeddings: List[List[float]],
                      metadata: Dict[str, Any]) -> List[str]:
    """Store tiles in Weaviate ISMA_Quantum collection."""
    stored_ids = []

    for tile, embedding in zip(tiles, embeddings):
        obj = {
            "class": "ISMA_Quantum",
            "properties": {
                "content": tile.text[:10000],
                "source_type": "chat_migration",
                "source_file": f"neo4j_7687_{metadata.get('conversation_id', 'unknown')}",
                "layer": 2,  # Application layer
                "priority": 0.6,  # Higher than corpus, we want chat data surfaced
                "phi_resonance": 0.6,
                "tile_index": tile.index,
                "start_char": tile.start_char,
                "end_char": tile.end_char,
                "token_count": tile.estimated_tokens,
                "checksum": compute_checksum(tile.text),
                "loaded_at": datetime.now().isoformat(),
                "timestamp": metadata.get("timestamp", datetime.now().isoformat()),
                "content_preview": tile.text[:500],
                "actor": "chat_migration",
                "conversation_id": metadata.get("conversation_id", ""),
                "exchange_index": metadata.get("exchange_index", 0),
            },
            "vector": embedding
        }

        try:
            response = requests.post(
                f"{WEAVIATE_URL}/v1/objects",
                json=obj,
                timeout=30
            )
            if response.status_code in [200, 201]:
                result = response.json()
                stored_ids.append(result.get("id", "unknown"))
            else:
                print(f"  Weaviate error: {response.status_code} - {response.text[:200]}")
        except Exception as e:
            print(f"  Error storing tile: {e}")

    return stored_ids


def query_exchanges(driver, skip: int = 0, limit: int = BATCH_SIZE) -> List[Dict[str, Any]]:
    """Query Exchange-Response pairs from legacy Neo4j."""
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Exchange)
            OPTIONAL MATCH (e)-[:HAS_RESPONSE]->(r:Response)
            WITH e, collect(r) as responses
            WHERE e.user_prompt IS NOT NULL AND size(e.user_prompt) > 10
            RETURN e.user_prompt as prompt,
                   e.timestamp as timestamp,
                   e.conversation_id as conversation_id,
                   e.index as exchange_index,
                   [r IN responses | r.text] as response_texts
            ORDER BY e.timestamp
            SKIP $skip
            LIMIT $limit
        """, skip=skip, limit=limit)

        return [dict(record) for record in result]


def query_chat_sessions(driver, skip: int = 0, limit: int = BATCH_SIZE) -> List[Dict[str, Any]]:
    """Query ChatSession-Message chains (newer structure)."""
    with driver.session() as session:
        result = session.run("""
            MATCH (cs:ChatSession)-[:HAS_MESSAGE]->(m:Message)
            WHERE m.content IS NOT NULL
            WITH cs, m
            ORDER BY m.createdAt
            WITH cs, collect({
                role: m.role,
                content: m.content,
                timestamp: m.createdAt
            }) as messages
            RETURN cs.platform as platform,
                   cs.purpose as purpose,
                   cs.url as url,
                   cs.created_at as session_created,
                   messages
            SKIP $skip
            LIMIT $limit
        """, skip=skip, limit=limit)

        return [dict(record) for record in result]


def format_exchange(exchange: Dict[str, Any]) -> str:
    """Format an exchange for embedding."""
    parts = []

    prompt = exchange.get("prompt", "")
    if prompt:
        parts.append(f"[User]: {prompt}")

    responses = exchange.get("response_texts", [])
    for resp in responses:
        if resp:
            parts.append(f"[Assistant]: {resp}")

    return "\n\n".join(parts)


def format_chat_session(session: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Format a chat session into exchange pairs for embedding."""
    exchanges = []
    messages = session.get("messages", [])

    current_exchange = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            if current_exchange:
                # Save previous exchange
                text = "\n\n".join(current_exchange)
                metadata = {
                    "platform": session.get("platform", ""),
                    "purpose": session.get("purpose", ""),
                    "conversation_id": session.get("url", "chat_session"),
                    "timestamp": session.get("session_created", ""),
                }
                exchanges.append((text, metadata))
                current_exchange = []
            current_exchange.append(f"[User]: {content}")
        elif role == "assistant":
            current_exchange.append(f"[Assistant]: {content}")

    # Don't forget last exchange
    if current_exchange:
        text = "\n\n".join(current_exchange)
        metadata = {
            "platform": session.get("platform", ""),
            "purpose": session.get("purpose", ""),
            "conversation_id": session.get("url", "chat_session"),
            "timestamp": session.get("session_created", ""),
        }
        exchanges.append((text, metadata))

    return exchanges


def count_exchanges(driver) -> int:
    """Count total exchanges to migrate."""
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Exchange)
            WHERE e.user_prompt IS NOT NULL AND size(e.user_prompt) > 10
            RETURN count(e) as count
        """)
        return result.single()["count"]


def count_chat_sessions(driver) -> int:
    """Count total chat sessions to migrate."""
    with driver.session() as session:
        result = session.run("""
            MATCH (cs:ChatSession)
            RETURN count(cs) as count
        """)
        return result.single()["count"]


def migrate_exchanges(driver, limit: int = None, start_from: int = 0):
    """Migrate Exchange-Response pairs."""
    total = count_exchanges(driver)
    print(f"\n{'='*60}")
    print(f"Migrating {total} exchanges from Neo4j 7687")
    print(f"{'='*60}")

    if limit:
        total = min(total, start_from + limit)

    migrated = 0
    tiles_created = 0
    tokens_processed = 0
    skip = start_from

    while skip < total:
        batch = query_exchanges(driver, skip=skip, limit=BATCH_SIZE)
        if not batch:
            break

        print(f"\n[Batch {skip//BATCH_SIZE + 1}] Processing {len(batch)} exchanges...")

        for i, exchange in enumerate(batch):
            text = format_exchange(exchange)
            if not text.strip() or len(text) < 50:
                continue

            # Tile the exchange
            tiles = phi_tile_text(text, f"exchange_{skip + i}", "chat")
            if not tiles:
                continue

            # Get embeddings
            texts = [t.text for t in tiles]
            embeddings = get_embeddings(texts)

            if embeddings is None:
                print(f"  Skipping exchange {skip + i} - embedding failed")
                continue

            # Prepare metadata
            metadata = {
                "conversation_id": exchange.get("conversation_id", f"batch_{skip}"),
                "timestamp": exchange.get("timestamp", ""),
                "exchange_index": exchange.get("exchange_index", i),
            }

            # Store in Weaviate
            stored_ids = store_in_weaviate(tiles, embeddings, metadata)

            if stored_ids:
                migrated += 1
                tiles_created += len(tiles)
                tokens_processed += sum(t.estimated_tokens for t in tiles)

        skip += BATCH_SIZE
        print(f"  Progress: {min(skip, total)}/{total} ({migrated} migrated, {tiles_created} tiles, {tokens_processed:,} tokens)")

    return migrated, tiles_created, tokens_processed


def migrate_chat_sessions(driver, limit: int = None, start_from: int = 0):
    """Migrate ChatSession-Message chains."""
    total = count_chat_sessions(driver)
    print(f"\n{'='*60}")
    print(f"Migrating {total} chat sessions from Neo4j 7687")
    print(f"{'='*60}")

    if limit:
        total = min(total, start_from + limit)

    migrated = 0
    tiles_created = 0
    tokens_processed = 0
    skip = start_from

    while skip < total:
        batch = query_chat_sessions(driver, skip=skip, limit=BATCH_SIZE)
        if not batch:
            break

        print(f"\n[Batch {skip//BATCH_SIZE + 1}] Processing {len(batch)} chat sessions...")

        for session in batch:
            exchanges = format_chat_session(session)

            for text, metadata in exchanges:
                if not text.strip() or len(text) < 50:
                    continue

                # Tile the exchange
                tiles = phi_tile_text(text, f"chat_{metadata.get('platform', 'unknown')}", "chat")
                if not tiles:
                    continue

                # Get embeddings
                texts = [t.text for t in tiles]
                embeddings = get_embeddings(texts)

                if embeddings is None:
                    continue

                # Store in Weaviate
                stored_ids = store_in_weaviate(tiles, embeddings, metadata)

                if stored_ids:
                    migrated += 1
                    tiles_created += len(tiles)
                    tokens_processed += sum(t.estimated_tokens for t in tiles)

        skip += BATCH_SIZE
        print(f"  Progress: {min(skip, total)}/{total} sessions ({migrated} exchanges migrated)")

    return migrated, tiles_created, tokens_processed


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Migrate chat data from Neo4j 7687 to ISMA")
    parser.add_argument("--limit", type=int, help="Max items to migrate")
    parser.add_argument("--start", type=int, default=0, help="Start from item N")
    parser.add_argument("--type", choices=["exchanges", "sessions", "all"], default="all",
                       help="What to migrate")
    parser.add_argument("--check", action="store_true", help="Check counts only")
    args = parser.parse_args()

    driver = get_neo4j_driver()

    try:
        if args.check:
            print("Checking Neo4j 7687 chat data...")
            exchanges = count_exchanges(driver)
            sessions = count_chat_sessions(driver)
            print(f"  Exchanges with prompts: {exchanges:,}")
            print(f"  Chat sessions: {sessions}")
            return

        total_migrated = 0
        total_tiles = 0
        total_tokens = 0

        if args.type in ["exchanges", "all"]:
            m, t, tok = migrate_exchanges(driver, limit=args.limit, start_from=args.start)
            total_migrated += m
            total_tiles += t
            total_tokens += tok

        if args.type in ["sessions", "all"]:
            m, t, tok = migrate_chat_sessions(driver, limit=args.limit, start_from=args.start)
            total_migrated += m
            total_tiles += t
            total_tokens += tok

        print(f"\n{'='*60}")
        print("MIGRATION COMPLETE")
        print(f"{'='*60}")
        print(f"  Exchanges migrated: {total_migrated:,}")
        print(f"  Tiles created: {total_tiles:,}")
        print(f"  Tokens processed: {total_tokens:,}")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
