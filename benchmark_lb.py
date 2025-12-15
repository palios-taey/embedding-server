#!/usr/bin/env python3
"""
Benchmark the embedding load balancer across 2 DGX Sparks (11 instances total).
Tests combined throughput and request distribution.
"""
import requests
import time
import concurrent.futures

LB_URL = "http://localhost:8090"

# Test sentences
sentences = [
    "Hello world",
    "This is a test sentence for embedding generation",
    "The quick brown fox jumps over the lazy dog near the river bank on a sunny afternoon",
    "Machine learning models can process natural language and generate high-dimensional vector representations",
] * 16  # 64 sentences per request

def embed_batch(texts, batch_size=64):
    """Send embedding request via load balancer."""
    response = requests.post(f"{LB_URL}/embed", json={"texts": texts, "batch_size": batch_size})
    return response.json()

def parallel_benchmark(num_parallel_requests, num_iterations=3):
    """Benchmark with parallel requests through load balancer."""
    total_tokens = 0
    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_requests) as executor:
        for _ in range(num_iterations):
            futures = [executor.submit(embed_batch, sentences) for _ in range(num_parallel_requests)]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                total_tokens += result["tokens_processed"]

    elapsed = time.time() - start
    throughput = total_tokens / elapsed

    return {
        "parallel_requests": num_parallel_requests,
        "iterations": num_iterations,
        "total_requests": num_parallel_requests * num_iterations,
        "total_tokens": total_tokens,
        "elapsed_s": elapsed,
        "throughput": throughput
    }

if __name__ == "__main__":
    print("=" * 70)
    print("Embedding Load Balancer Benchmark")
    print("11 instances across 2 DGX Sparks (Spark 1: 6, Spark 2: 5)")
    print("=" * 70)

    # Warmup
    print("\nWarming up...")
    embed_batch(sentences[:4])

    # Test with increasing parallelism
    print("\n--- Throughput vs Parallelism ---")
    for parallel in [1, 2, 4, 8, 11, 22]:
        result = parallel_benchmark(parallel, num_iterations=3)
        print(f"  {parallel:2d} parallel: {result['throughput']:,.0f} tok/s "
              f"({result['total_requests']} requests in {result['elapsed_s']:.1f}s)")

    # Final full benchmark
    print("\n--- Full Benchmark (22 parallel, 5 iterations) ---")
    result = parallel_benchmark(22, num_iterations=5)
    print(f"Total requests: {result['total_requests']}")
    print(f"Total tokens: {result['total_tokens']:,}")
    print(f"Elapsed time: {result['elapsed_s']:.2f}s")
    print(f"Combined throughput: {result['throughput']:,.0f} tok/s")

    # Theoretical calculation
    print("\n--- Throughput Analysis ---")
    single_instance = 2500  # tok/s per instance (from earlier benchmark)
    spark1_instances = 6
    spark2_instances = 5
    total_instances = spark1_instances + spark2_instances

    # Each Spark has 1 GPU, so max throughput is 2 GPUs
    theoretical_max = single_instance * 2  # 2 GPUs = 2x throughput
    print(f"Single instance throughput: {single_instance:,} tok/s")
    print(f"Theoretical max (2 GPUs): {theoretical_max:,} tok/s")
    print(f"Actual throughput: {result['throughput']:,.0f} tok/s")
    print(f"Efficiency: {result['throughput']/theoretical_max*100:.1f}%")

    print("\n" + "=" * 70)
    print(f"RESULT: {result['throughput']:,.0f} tok/s across 2 DGX Sparks")
    print("=" * 70)
