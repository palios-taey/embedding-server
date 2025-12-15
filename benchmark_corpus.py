#!/usr/bin/env python3
"""
Benchmark embedding server with real corpus files.
Uses files from /home/spark/data/expansion_md/ for realistic workload.
"""
import requests
import time
import os
import random
import concurrent.futures
from pathlib import Path

LB_URL = "http://localhost:8090"
CORPUS_DIR = "/home/spark/data/expansion_md"

def get_sample_files(n=100, seed=42):
    """Get random sample of markdown files from corpus."""
    random.seed(seed)
    all_files = list(Path(CORPUS_DIR).rglob("*.md"))
    return random.sample(all_files, min(n, len(all_files)))

def read_file_content(filepath):
    """Read file content, handle encoding issues."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return ""

def embed_texts(texts, batch_size=64):
    """Send embedding request."""
    response = requests.post(
        f"{LB_URL}/embed",
        json={"texts": texts, "batch_size": batch_size},
        timeout=300
    )
    return response.json()

def benchmark_sequential(files, batch_size=64):
    """Sequential benchmark - process files in batches."""
    total_tokens = 0
    total_chars = 0
    batches_processed = 0

    # Read all file contents
    contents = [read_file_content(f) for f in files]
    contents = [c for c in contents if c.strip()]  # Filter empty

    print(f"Loaded {len(contents)} files with content")
    print(f"Total chars: {sum(len(c) for c in contents):,}")

    start = time.time()

    # Process in batches
    for i in range(0, len(contents), batch_size):
        batch = contents[i:i+batch_size]
        result = embed_texts(batch, batch_size=batch_size)
        total_tokens += result["tokens_processed"]
        total_chars += sum(len(t) for t in batch)
        batches_processed += 1

    elapsed = time.time() - start

    return {
        "mode": "sequential",
        "files": len(contents),
        "batches": batches_processed,
        "batch_size": batch_size,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "elapsed_s": elapsed,
        "throughput_tok_s": total_tokens / elapsed,
        "throughput_char_s": total_chars / elapsed,
    }

def benchmark_parallel(files, batch_size=64, parallel_requests=4):
    """Parallel benchmark - send multiple batches concurrently."""
    total_tokens = 0
    total_chars = 0

    # Read all file contents
    contents = [read_file_content(f) for f in files]
    contents = [c for c in contents if c.strip()]

    # Split into batches
    batches = [contents[i:i+batch_size] for i in range(0, len(contents), batch_size)]

    print(f"Processing {len(batches)} batches with {parallel_requests} parallel requests")

    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_requests) as executor:
        futures = [executor.submit(embed_texts, batch, batch_size) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            total_tokens += result["tokens_processed"]

    elapsed = time.time() - start
    total_chars = sum(len(c) for c in contents)

    return {
        "mode": "parallel",
        "parallel_requests": parallel_requests,
        "files": len(contents),
        "batches": len(batches),
        "batch_size": batch_size,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "elapsed_s": elapsed,
        "throughput_tok_s": total_tokens / elapsed,
        "throughput_char_s": total_chars / elapsed,
    }

def run_full_benchmark():
    """Run comprehensive benchmark suite."""
    print("=" * 70)
    print("Embedding Server Benchmark - Real Corpus Files")
    print("=" * 70)

    # Check LB health
    try:
        health = requests.get(f"{LB_URL}/health", timeout=5).json()
        print(f"LB Status: {health['status']}")
        print(f"Model: {health['model']}")
        print(f"GPU Memory: {health['memory_used_gb']:.1f} / {health['memory_total_gb']:.1f} GB")
    except Exception as e:
        print(f"ERROR: Cannot connect to LB at {LB_URL}: {e}")
        return

    # Get sample files
    print("\n--- Loading corpus files ---")
    files = get_sample_files(n=200, seed=42)
    print(f"Selected {len(files)} files for benchmark")

    # Warmup
    print("\n--- Warmup ---")
    warmup_content = [read_file_content(files[0])]
    embed_texts(warmup_content)
    print("Warmup complete")

    results = []

    # Sequential benchmark
    print("\n--- Sequential Benchmark ---")
    result = benchmark_sequential(files, batch_size=32)
    print(f"Throughput: {result['throughput_tok_s']:,.0f} tok/s")
    results.append(result)

    # Parallel benchmarks with varying concurrency
    print("\n--- Parallel Benchmarks ---")
    for parallel in [2, 4, 8]:
        result = benchmark_parallel(files, batch_size=32, parallel_requests=parallel)
        print(f"  {parallel} parallel: {result['throughput_tok_s']:,.0f} tok/s")
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        mode = r['mode']
        if mode == 'parallel':
            mode = f"parallel-{r['parallel_requests']}"
        print(f"  {mode:15s}: {r['throughput_tok_s']:>8,.0f} tok/s | {r['elapsed_s']:>6.1f}s | {r['total_tokens']:>8,} tokens")

    print("\n" + "=" * 70)
    best = max(results, key=lambda x: x['throughput_tok_s'])
    print(f"BEST: {best['throughput_tok_s']:,.0f} tok/s")
    print("=" * 70)

    return results

if __name__ == "__main__":
    run_full_benchmark()
