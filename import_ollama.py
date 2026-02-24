import ollama
import time
import psutil
import pandas as pd
import threading
from pynvml import *

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

gpu_usage_list = []
vram_usage_list = []

monitoring = True

def monitor_gpu():
    while monitoring:
        util = nvmlDeviceGetUtilizationRates(handle)
        mem = nvmlDeviceGetMemoryInfo(handle)

        gpu_usage_list.append(util.gpu)
        vram_usage_list.append((mem.used / mem.total) * 100)

        time.sleep(0.1)

models = ["llama3", "mistral", "phi3"]

prompt = input("Enter prompt: ")

results = []

for model in models:

    print(f"\nAnalyzing {model}...")

    gpu_usage_list.clear()
    vram_usage_list.clear()
    monitoring = True

    monitor_thread = threading.Thread(target=monitor_gpu)
    monitor_thread.start()

    start = time.time()

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    end = time.time()

    monitoring = False
    monitor_thread.join()

    avg_gpu = sum(gpu_usage_list)/len(gpu_usage_list)
    max_gpu = max(gpu_usage_list)

    avg_vram = sum(vram_usage_list)/len(vram_usage_list)
    max_vram = max(vram_usage_list)

    time_taken = end - start

    bottleneck = "None"

    if max_gpu < 10 and max_vram > 20:
        bottleneck = "GPU underutilized"
    elif max_vram > 90:
        bottleneck = "VRAM bottleneck"
    elif time_taken > 30:
        bottleneck = "Model too slow"

    print(f"Time: {round(time_taken,2)} sec")
    print(f"Avg GPU: {round(avg_gpu,2)}%")
    print(f"Max GPU: {round(max_gpu,2)}%")
    print(f"Max VRAM: {round(max_vram,2)}%")
    print(f"Bottleneck: {bottleneck}")

    results.append({
        "Model": model,
        "Time": time_taken,
        "Avg_GPU": avg_gpu,
        "Max_GPU": max_gpu,
        "Max_VRAM": max_vram,
        "Bottleneck": bottleneck
    })

df = pd.DataFrame(results)

print("\nResults:")
print(df)

nvmlShutdown()