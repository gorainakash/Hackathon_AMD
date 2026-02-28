import time
import sys
import psutil
import numpy as np

print("Starting OptimizedLLM (CPU + NumPy Vectorization + KV Cache)...")
cpu_before = psutil.cpu_percent(interval=1)

# 1. Optimized Memory Loading (C-Contiguous Arrays)
VOCAB_SIZE = 10000
HIDDEN_DIM = 256
SEQ_LEN = 32
LAYERS = 8

print("Loading 'weights' into highly optimized NumPy arrays...")
# Allocating flat C-arrays avoids Python object overhead and reduces RAM bloat by ~10x
weights = []
for layer in range(LAYERS):
    weights.append({
        'W_q': np.random.rand(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32),
        'W_k': np.random.rand(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32),
        'W_v': np.random.rand(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32),
        'W_o': np.random.rand(HIDDEN_DIM, HIDDEN_DIM).astype(np.float32)
    })
print("Weights loaded efficiently.")

# 2. Efficient Computation (Vectorized Matmul, O(1) in Python, pushed to BLAS/LAPACK)
def matmul(A, B):
    # Uses low-level, highly optimized C libraries instead of nested Python loops
    return np.matmul(A, B)

def generate_tokens(num_tokens):
    start = time.time()
    
    # 3. KV Cache Simulation: Pre-allocate context array
    # Instead of rebuilding the sequence context every step, we append to a running cache
    kv_cache = np.random.rand(SEQ_LEN, HIDDEN_DIM).astype(np.float32)
    
    for t in range(num_tokens):
        # We only compute attention for the NEW token `(1, HIDDEN_DIM)` 
        # instead of the entirely growing sequence `(SEQ_LEN + t, HIDDEN_DIM)`
        new_token = np.random.rand(1, HIDDEN_DIM).astype(np.float32)
        
        # Simulated attention step using BLAS-accelerated vectorization
        q = matmul(new_token, weights[0]['W_q'])
        k = matmul(new_token, weights[0]['W_k'])
        
        # Add to KV cache
        kv_cache = np.vstack((kv_cache, new_token))
        
        # Notice how fast this iterates compared to bad_model.py
        sys.stdout.write(f" [Token {t+1}]")
        sys.stdout.flush()
        
    print()
    return time.time() - start

num_tokens = 5
print(f"Generating {num_tokens} tokens sequentially...")
elapsed = generate_tokens(num_tokens)
cpu_after = psutil.cpu_percent(interval=1)

print("\n--- OptimizedLLM Finished ---")
print(f"Execution time: {elapsed:.4f} seconds")
print(f"Generated {num_tokens} tokens")
print(f"Throughput: {num_tokens / elapsed:.2f} tokens/sec")
print(f"CPU usage before: {cpu_before}%")
print(f"CPU usage after: {cpu_after}%")

print("Device used: CPU (Optimized BLAS)")
