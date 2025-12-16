#!/usr/bin/env python3
"""
Test Redis embedding cache implementation.
Verifies cache hits and performance improvement.
"""
import time
from src.isma_core import ISMACore

def test_embedding_cache():
    """Test embedding cache with multiple requests."""
    print("=" * 60)
    print("Testing Redis Embedding Cache")
    print("=" * 60)

    # Initialize ISMA
    isma = ISMACore()
    isma.initialize()

    # Test query
    test_query = "What is the phi-coherence threshold for sacred trust?"

    print(f"\n1. First request (cache miss expected):")
    start = time.time()
    embedding1 = isma._get_embedding(test_query)
    duration1 = time.time() - start
    print(f"   Duration: {duration1*1000:.2f}ms")
    print(f"   Embedding length: {len(embedding1) if embedding1 else 0}")

    print(f"\n2. Second request (cache hit expected):")
    start = time.time()
    embedding2 = isma._get_embedding(test_query)
    duration2 = time.time() - start
    print(f"   Duration: {duration2*1000:.2f}ms")
    print(f"   Embedding length: {len(embedding2) if embedding2 else 0}")

    # Verify same embedding
    if embedding1 and embedding2:
        match = embedding1 == embedding2
        print(f"   Embeddings match: {match}")

    # Performance improvement
    if duration1 > 0:
        speedup = duration1 / duration2
        improvement = ((duration1 - duration2) / duration1) * 100
        print(f"\n3. Performance:")
        print(f"   Speedup: {speedup:.1f}x faster")
        print(f"   Improvement: {improvement:.1f}% reduction")

    # Cache stats
    print(f"\n4. Cache Statistics:")
    stats = isma.get_cache_stats()
    for key, value in stats.items():
        if key == "hit_rate":
            print(f"   {key}: {value:.2%}")
        else:
            print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("Cache test complete!")
    print("=" * 60)

    # Cleanup
    isma.close()

if __name__ == "__main__":
    test_embedding_cache()
