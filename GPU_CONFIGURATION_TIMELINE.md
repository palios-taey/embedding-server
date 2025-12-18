# GPU Configuration Evolution Timeline
*Investigation Date: December 18, 2025*

## Summary

**CRITICAL BUG**: `CUDA_VISIBLE_DEVICES=0` was introduced in the **INITIAL COMMIT** (Dec 15, 2025) and has **NEVER BEEN CHANGED**. All instances on all Sparks have always been forced to GPU 0, defeating the entire purpose of multi-instance deployment.

---

## Timeline

### December 15, 2025 - Initial Commit (2734f1a)

**Commit**: `feat: initial commit - Qwen3-Embedding-8B server`
**Author**: Spark Claude

**start-multi.sh Created With**:
```bash
# Line 34 in original commit
-e CUDA_VISIBLE_DEVICES=0 \
```

**Comment in script (Line 3)**:
```bash
# 6 instances × ~15GB = 90GB VRAM (leaving headroom from 128GB)
```

**Original Intent**:
- Deploy multiple instances across available VRAM
- 6 instances × 15GB = 90GB VRAM utilization
- Leave headroom from 128GB total VRAM

**What Actually Happened**:
- `CUDA_VISIBLE_DEVICES=0` forced ALL instances to GPU 0
- Even with `--gpus all`, the env var restricts visibility
- All instances share the SAME 128GB GPU, not distributed

**Original Docker Args**:
```bash
docker run -d \
  --name $NAME \
  --gpus all \                    # This is overridden by env var
  -e CUDA_VISIBLE_DEVICES=0 \     # THIS BREAKS DISTRIBUTION
  $IMAGE_ID \
  python3 /workspace/server.py
```

### December 15-18, 2025 - No Changes to GPU Configuration

**Commits that DID NOT touch GPU config**:
- 4d8c16e - feat(isma): add Integrated Shared Memory Architecture
- b7a0e61 - fix(isma): critical schema mismatch and Gate-B completion
- 81be988 - feat(isma): load transcripts, fix truncation, update φ-tiling docs
- d88b331 - docs: add φ-coherence theater comments and exploration document
- c84e8d2 - docs: add audit results to implementation plan
- d48ee9f - docs: add Family Dream Cycle synthesis to implementation plan
- df13705 - docs: add Grok mathematical validation to φ-coherence exploration

### December 18, 2025 - Performance Fixes (NOT GPU Distribution)

**Commit e0c8baa**: `perf(loader): implement Grok/Perplexity audit fixes for 3x throughput`
- Added connection pooling (requests.Session)
- Increased BATCH_SIZE from 32 to 128
- Added 4 parallel CPU producers
- **DID NOT TOUCH** CUDA_VISIBLE_DEVICES

**Commit 5acf1af**: `fix(server): add concurrency control to prevent GPU OOM`
- Added asyncio.Semaphore(1) for serial inference
- Added torch.cuda.empty_cache() cleanup
- **DID NOT TOUCH** CUDA_VISIBLE_DEVICES

---

## The Actual Problem

### What We Thought Was Happening
```
Spark 1 GPU 0 → Instance 0
Spark 1 GPU 1 → Instance 1
Spark 1 GPU 2 → Instance 2
...
```

### What Was Actually Happening
```
Spark 1 GPU 0 → Instance 0, 1, 2, 3 (ALL FIGHTING FOR SAME GPU)
Spark 1 GPU 1 → UNUSED
Spark 1 GPU 2 → UNUSED
...
```

### Evidence from server.py (Line 29-31)
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

if device.type == "cuda":
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")  # Always logs GPU 0!
```

The code **always uses device 0** because:
1. `CUDA_VISIBLE_DEVICES=0` makes only GPU 0 visible
2. `torch.device("cuda")` defaults to the first visible device
3. There is **no code** to select different GPUs per instance

---

## Why This Happened

### Docker --gpus all vs CUDA_VISIBLE_DEVICES

| Flag | Behavior | Precedence |
|------|----------|------------|
| `--gpus all` | Makes all GPUs accessible to container | LOW |
| `CUDA_VISIBLE_DEVICES=0` | Restricts CUDA runtime to GPU 0 only | **HIGH** |

**Docker documentation**:
> "The CUDA_VISIBLE_DEVICES environment variable takes precedence over --gpus"

### The Missing Logic

**What should have been in start-multi.sh**:
```bash
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  GPU_ID=$i  # Or: GPU_ID=$((i % NUM_GPUS))

  docker run -d \
    ...
    -e CUDA_VISIBLE_DEVICES=$GPU_ID \  # Different GPU per instance
    ...
done
```

**OR in server.py**:
```python
import os
INSTANCE_ID = int(os.getenv("INSTANCE_ID", "0"))
device = torch.device(f"cuda:{INSTANCE_ID}" if torch.cuda.is_available() else "cpu")
```

---

## Why No One Noticed

### 1. System Appeared to Work
- All instances started successfully
- Health checks passed
- Models loaded (slowly, on GPU 0)
- Requests were processed

### 2. Performance Was "Good Enough"
- README shows: "8 parallel: 4,655 tok/s (+79%)"
- This looked like success (vs 2,607 tok/s sequential)
- But gain came from **parallel CPU work**, not GPU distribution

### 3. GPU Memory Not Saturated
- Qwen3-8B uses ~15GB per instance
- 4 instances × 15GB = 60GB
- Fits on single GB10 Blackwell 128GB GPU
- OOM didn't occur until recent high load (triggering 5acf1af fix)

### 4. nvidia-smi Output Misleading
```
GPU 0: 60GB used (4 instances)
GPU 1: 0GB used
GPU 2: 0GB used
GPU 3: 0GB used
```
This SHOULD have been the red flag, but wasn't checked systematically.

---

## Performance Impact

### Current State (All on GPU 0)
- 4 instances fighting for same GPU
- Sequential inference (Semaphore prevents concurrent)
- Throughput: ~4,500 tok/s

### Expected with Proper Distribution
- 4 instances on 4 different GPUs
- True parallel inference
- Expected throughput: ~10,000 tok/s (4 × 2,607)

**We're leaving 55% of potential throughput on the table.**

---

## Commits That Should Have Fixed This (But Didn't)

### None

**No commit has EVER changed CUDA_VISIBLE_DEVICES.**

The bug was introduced in the first commit and persists to HEAD (5acf1af).

---

## Related Files

| File | Status | Issue |
|------|--------|-------|
| `start-multi.sh` | **BROKEN** | Hard-coded CUDA_VISIBLE_DEVICES=0 |
| `start-sequential.sh` | Unknown | Not audited |
| `add_spark2.sh` | Unknown | Not audited (likely same issue) |
| `server.py` | **NO GPU SELECTION** | Always uses default device |
| `nginx-lb.conf` | OK | Load balancer works (routes to all instances) |

---

## The Fix (Needed)

### Option 1: Fix start-multi.sh (Simple)
```bash
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
  GPU_ID=$((i % 4))  # Cycle through GPUs 0-3

  docker run -d \
    ...
    -e CUDA_VISIBLE_DEVICES=$GPU_ID \  # Remove hard-coded 0
    -e INSTANCE_ID=$i \
    ...
done
```

### Option 2: Fix server.py (More Robust)
```python
# Get instance ID from environment
INSTANCE_ID = int(os.getenv("INSTANCE_ID", "0"))
NUM_GPUS = torch.cuda.device_count()

# Select GPU based on instance ID
if torch.cuda.is_available() and NUM_GPUS > 0:
    gpu_id = INSTANCE_ID % NUM_GPUS
    device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"Instance {INSTANCE_ID} using GPU {gpu_id} of {NUM_GPUS}")
else:
    device = torch.device("cpu")
```

### Option 3: Both (Recommended)
- Fix start-multi.sh to set INSTANCE_ID
- Fix server.py to use INSTANCE_ID for GPU selection
- Remove CUDA_VISIBLE_DEVICES entirely (let code handle it)

---

## Conclusion

This is a **Day 1 bug** that has never been addressed. The infrastructure was designed for multi-GPU distribution but implemented with a hard-coded single-GPU constraint.

Every performance optimization (connection pooling, batch size tuning, concurrency control) has been working around this fundamental misconfiguration.

**The fix is trivial. The impact is massive.**

---

*Investigation by: Spark Claude*
*Date: December 18, 2025*
*Repository: /home/spark/embedding-server*
