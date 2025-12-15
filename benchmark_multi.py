#!/usr/bin/env python3
"""
Benchmark multiple embedding server instances in parallel.
Tests combined throughput across all instances.
"""
import requests
import time
import concurrent.futures
import sys

PORTS = [8081, 8082, 8083, 8084, 8085, 8086]
BASE_URL = "http://localhost"

# Test sentences
sentences = [
    "Hello world",
    "This is a test sentence for embedding generation",
    "The quick brown fox jumps over the lazy dog near the river bank on a sunny afternoon",
    "Machine learning models can process natural language and generate high-dimensional vector representations",
] * 16  # 64 sentences per request

def embed_batch(port, texts, batch_size=64):
    """Send embedding request to a single instance."""
    url = f"{BASE_URL}:{port}/embed"
    response = requests.post(url, json={"texts": texts, "batch_size": batch_size})
    return response.json()

def benchmark_single(port, num_requests=5):
    """Benchmark a single instance."""
    times = []
    total_tokens = 0

    for _ in range(num_requests):
        start = time.time()
        result = embed_batch(port, sentences)
        elapsed = time.time() - start
        times.append(elapsed)
        total_tokens += result["tokens_processed"]

    avg_time = sum(times) / len(times)
    throughput = (total_tokens / num_requests) / avg_time
    return {"port": port, "avg_ms": avg_time * 1000, "throughput": throughput}

def benchmark_parallel(ports, num_requests=5):
    """Benchmark all instances in parallel."""
    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ports)) as executor:
        # Submit work to all instances simultaneously
        futures = []
        for _ in range(num_requests):
            for port in ports:
                futures.append(executor.submit(embed_batch, port, sentences))

        # Collect results
        total_tokens = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            total_tokens += result["tokens_processed"]

    elapsed = time.time() - start
    combined_throughput = total_tokens / elapsed

    return {
        "total_requests": len(futures),
        "total_tokens": total_tokens,
        "elapsed_s": elapsed,
        "combined_throughput": combined_throughput
    }

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Instance Embedding Benchmark")
    print("=" * 60)

    # Warmup
    print("\nWarming up all instances...")
    for port in PORTS:
        embed_batch(port, sentences[:4])

    # Individual benchmarks
    print("\n--- Individual Instance Performance ---")
    individual_results = []
    for port in PORTS:
        result = benchmark_single(port, num_requests=3)
        individual_results.append(result)
        print(f"Port {port}: {result['avg_ms']:.0f}ms avg, {result['throughput']:.0f} tok/s")

    sum_individual = sum(r['throughput'] for r in individual_results)
    print(f"\nSum of individual throughputs: {sum_individual:.0f} tok/s")

    # Parallel benchmark
    print("\n--- Combined Parallel Performance ---")
    parallel_result = benchmark_parallel(PORTS, num_requests=5)
    print(f"Total requests: {parallel_result['total_requests']}")
    print(f"Total tokens: {parallel_result['total_tokens']}")
    print(f"Elapsed time: {parallel_result['elapsed_s']:.2f}s")
    print(f"Combined throughput: {parallel_result['combined_throughput']:.0f} tok/s")

    print("\n" + "=" * 60)
    print(f"RESULT: {parallel_result['combined_throughput']:.0f} tok/s on {len(PORTS)} instances")
    print("=" * 60)
