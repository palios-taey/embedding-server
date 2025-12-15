import requests
import time
import json

URL = "http://localhost:8081/embed"

# Test sentences of varying lengths
sentences = [
    "Hello world",
    "This is a test sentence for embedding generation",
    "The quick brown fox jumps over the lazy dog near the river bank on a sunny afternoon",
    "Machine learning models can process natural language and generate high-dimensional vector representations",
] * 16  # 64 sentences total

print(f"Testing with {len(sentences)} sentences...")

# Warmup
requests.post(URL, json={"texts": sentences[:4]})

# Benchmark
results = []
for batch_size in [8, 16, 32, 64]:
    batch = sentences[:batch_size]

    times = []
    for _ in range(5):
        start = time.time()
        response = requests.post(URL, json={"texts": batch, "batch_size": batch_size})
        elapsed = time.time() - start
        times.append(elapsed)
        data = response.json()

    avg_time = sum(times) / len(times)
    tokens = data["tokens_processed"]
    throughput = tokens / avg_time

    print(f"Batch {batch_size:2d}: {avg_time*1000:.0f}ms avg, {tokens} tokens, {throughput:.0f} tok/s")
    results.append({"batch_size": batch_size, "avg_ms": avg_time*1000, "tokens": tokens, "throughput": throughput})

print("\nBest throughput:", max(results, key=lambda x: x["throughput"]))
