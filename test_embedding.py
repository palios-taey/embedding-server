import torch
import time
from transformers import AutoModel, AutoTokenizer

print("Loading Qwen3-Embedding-8B...")
start = time.time()

model_name = "Qwen/Qwen3-Embedding-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.cuda()
model.eval()

load_time = time.time() - start
print(f"Model loaded in {load_time:.1f}s")
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# Test embedding
test_texts = ["Hello world", "This is a test sentence for embedding generation"]
print(f"\nGenerating embeddings for {len(test_texts)} texts...")

start = time.time()
with torch.no_grad():
    inputs = tokenizer(test_texts, padding=True, truncation=True, max_length=4096, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    # Mean pooling over sequence dimension
    embeddings = outputs.last_hidden_state.mean(dim=1)

embed_time = time.time() - start
print(f"Embedding shape: {embeddings.shape}")
print(f"Embedding time: {embed_time*1000:.1f}ms")
print(f"Embedding dim: {embeddings.shape[-1]}")
print("\nSuccess! Embedding server is feasible.")
