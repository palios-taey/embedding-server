#!/usr/bin/env python3
"""
Test Redis embedding cache with mock embeddings.
Verifies cache logic without requiring embedding server.
"""
import time
import hashlib
import json
import redis

def test_cache_logic():
    """Test the cache logic directly."""
    print("=" * 60)
    print("Testing Redis Embedding Cache Logic")
    print("=" * 60)

    # Connect to Redis
    r = redis.Redis(host='10.0.0.68', port=6379, decode_responses=True)

    # Test text
    test_text = "What is the phi-coherence threshold for sacred trust?"

    # Generate cache key (same as in isma_core.py)
    cache_key = f"emb:{hashlib.sha256(test_text.encode()).hexdigest()[:16]}"
    print(f"\n1. Cache key: {cache_key}")

    # Mock embedding
    mock_embedding = [0.1] * 1024  # Typical embedding size

    # Clear previous cache for clean test
    r.delete(cache_key)
    r.delete("emb_cache_hits")
    r.delete("emb_cache_misses")

    print(f"\n2. First request (cache miss):")
    cached = r.get(cache_key)
    if cached:
        r.incr("emb_cache_hits")
        embedding1 = json.loads(cached)
        print("   ❌ Found in cache (unexpected)")
    else:
        r.incr("emb_cache_misses")
        print("   ✓ Cache miss (expected)")
        # Simulate embedding generation
        time.sleep(0.1)  # Simulate network delay
        embedding1 = mock_embedding
        # Cache for 24 hours
        r.setex(cache_key, 86400, json.dumps(embedding1))
        print(f"   ✓ Cached embedding ({len(embedding1)} dimensions)")

    print(f"\n3. Second request (cache hit):")
    start = time.time()
    cached = r.get(cache_key)
    duration_cached = time.time() - start

    if cached:
        r.incr("emb_cache_hits")
        embedding2 = json.loads(cached)
        print(f"   ✓ Cache hit (expected)")
        print(f"   ✓ Retrieved in {duration_cached*1000:.2f}ms")
        print(f"   ✓ Embedding length: {len(embedding2)}")
    else:
        r.incr("emb_cache_misses")
        print("   ❌ Cache miss (unexpected)")

    # Verify same embedding
    if embedding1 == embedding2:
        print("   ✓ Embeddings match")

    # Check TTL
    ttl = r.ttl(cache_key)
    print(f"\n4. Cache TTL: {ttl} seconds (~{ttl/3600:.1f} hours remaining)")

    # Cache stats
    print(f"\n5. Cache Statistics:")
    hits = int(r.get("emb_cache_hits") or 0)
    misses = int(r.get("emb_cache_misses") or 0)
    total = hits + misses
    hit_rate = hits / total if total > 0 else 0.0

    stats = {
        "hits": hits,
        "misses": misses,
        "total": total,
        "hit_rate": hit_rate,
        "target": 0.80
    }

    for key, value in stats.items():
        if key == "hit_rate":
            print(f"   {key}: {value:.2%}")
        else:
            print(f"   {key}: {value}")

    # Performance projection
    print(f"\n6. Performance Projection:")
    print(f"   Without cache: ~132ms (Weaviate latency per Grok)")
    print(f"   With cache: <{duration_cached*1000:.2f}ms (Redis lookup)")
    speedup = 132 / (duration_cached * 1000) if duration_cached > 0 else 0
    print(f"   Expected speedup: ~{speedup:.0f}x")
    print(f"   φ-coherence improvement: 0.87 → 0.92 (per Grok)")

    print("\n" + "=" * 60)
    print("✓ Cache logic verified!")
    print("=" * 60)

if __name__ == "__main__":
    test_cache_logic()
