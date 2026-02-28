import time
import sys
import psutil

print("Starting UnoptimizedLLM (CPU-only, no vectorization, no KV cache)...")
cpu_before = psutil.cpu_percent(interval=1)

# 1. Unoptimized Memory Loading (Memory Bloat)
VOCAB_SIZE = 10000
HIDDEN_DIM = 256
SEQ_LEN = 32
LAYERS = 8

def create_random_matrix(rows, cols):
    # Pure Python lists, causing high overhead instead of using flat C-arrays (numpy/tensors)
    return [[(r * c) % 100 / 100.0 for c in range(cols)] for r in range(rows)]

print("Loading 'weights' into explicit Python Lists...")
weights = []
for layer in range(LAYERS):
    weights.append({
        'W_q': create_random_matrix(HIDDEN_DIM, HIDDEN_DIM),
        'W_k': create_random_matrix(HIDDEN_DIM, HIDDEN_DIM),
        'W_v': create_random_matrix(HIDDEN_DIM, HIDDEN_DIM),
        'W_o': create_random_matrix(HIDDEN_DIM, HIDDEN_DIM)
    })
print("Weights loaded. Warning: High RAM usage due to Python object overhead.")

# 2. Inefficient Computation (No Vectorization, O(N^3) CPU load)
def matmul(A, B):
    # O(N^3) Python nested loops. Extremely slow and CPU intensive.
    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])
    
    C = [[0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for k in range(cols_A):
            for j in range(cols_B):
                C[i][j] += A[i][k] * B[k][j]
    return C

def generate_tokens(num_tokens):
    start = time.time()
    for t in range(num_tokens):
        # 3. No KV Cache: Recomputes context for entire sequence every time
        input_state = create_random_matrix(SEQ_LEN + t, HIDDEN_DIM)
        
        # Simulate ONE layer's attention bottleneck to freeze the CPU natively
        q = matmul(input_state, weights[0]['W_q'])
        k = matmul(input_state, weights[0]['W_k'])
        
        sys.stdout.write(f" [Token {t+1}]")
        sys.stdout.flush()
        
    print()
    return time.time() - start

num_tokens = 5
print(f"Generating {num_tokens} tokens sequentially...")
elapsed = generate_tokens(num_tokens)
cpu_after = psutil.cpu_percent(interval=1)

print("\n--- UnoptimizedLLM Finished ---")
print(f"Execution time: {elapsed:.2f} seconds")
print(f"Generated {num_tokens} tokens")
print(f"Throughput: {num_tokens / elapsed:.2f} tokens/sec")
print(f"CPU usage before: {cpu_before}%")
print(f"CPU usage after: {cpu_after}%")

print("Device used: CPU")
