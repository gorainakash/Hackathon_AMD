import streamlit as st
import ollama
import time
import psutil
import pandas as pd
import threading
import concurrent.futures
import subprocess
from pynvml import *

# --- Setup Streamlit page ---
st.set_page_config(page_title="ConcurML: Resource Profiler", layout="wide", page_icon="⚙️")

# Custom CSS for better UI
st.markdown("""
    <style>
    .big-font { font-size: 24px !important; font-weight: bold; }
    .stMetric { background-color: #1a1c24; padding: 15px; border-radius: 12px; border-left: 5px solid #ff4b4b; }
    .stMetric:nth-child(even) { border-left-color: #00ffcc; }
    </style>
""", unsafe_allow_html=True)

st.title("🧩 ConcurML: Optimization Profiler")
st.markdown("**Core Objective**: Visualize hardware resource usage for *concurrent* ML models and **demonstrate targeted performance optimizations**.")

# --- System API Initialization ---
@st.cache_resource
def init_gpu_monitoring():
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        return handle, True
    except NVMLError:
        return None, False

handle, has_gpu = init_gpu_monitoring()

# --- Sidebar ---
st.sidebar.header("⚙️ Execution Setting")

available_models = ["llama3", "mistral", "phi3", "bad_model", "optimized_model"]
selected_models = st.sidebar.multiselect(
    "Select Models to Run Concurrently", 
    available_models, 
    default=["bad_model", "optimized_model"] # Default to proving the optimization
)

prompt_text = st.sidebar.text_area("LLM Prompt", "Explain the architecture of an operating system kernel in 5 paragraphs.")

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ VRAM Guardian (Smart Scheduler)")

if has_gpu:
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    total_vram_gb = mem_info.total / (1024**3)
    st.sidebar.write(f"Total VRAM Available: **{total_vram_gb:.2f} GB**")

    # Rough estimates
    vram_estimates = {"llama3": 4.5, "mistral": 4.5, "phi3": 2.5, "bad_model": 0, "optimized_model": 0}
    est_vram = sum([vram_estimates.get(m, 0) for m in selected_models])
    
    if est_vram > total_vram_gb * 0.9:
        st.sidebar.error(f"⚠️ **Warning**: Estimated VRAM ({est_vram:.1f} GB) might exceed available. Expect OOM or heavy swapping!")
    else:
        st.sidebar.success(f"✅ Safe: Estimated VRAM footprint is {est_vram:.1f} GB.")
else:
    st.sidebar.warning("No NVIDIA GPU detected. Using CPU only.")

# --- Background Task Definitions ---
def run_ollama_model(model_name, prmpt):
    try:
        start_time = time.time()
        stream = ollama.chat(model=model_name, messages=[{"role": "user", "content": prmpt}], stream=True)
        
        first_token_time = None
        token_count = 0
        
        for chunk in stream:
            if first_token_time is None:
                first_token_time = time.time() - start_time
            token_count += 1
            
        end_time = time.time()
        total_time = end_time - start_time
        tps = token_count / (total_time - first_token_time) if first_token_time and total_time > first_token_time else 0
        
        return {
            "Model": model_name,
            "Status": "✅ Success",
            "Total Time (s)": round(total_time, 2),
            "TTFT (s)": round(first_token_time, 2) if first_token_time else 0,
            "Tokens": token_count,
            "Tokens/sec": round(tps, 2)
        }
    except Exception as e:
        return {"Model": model_name, "Status": f"❌ Error: {e}", "Total Time (s)": 0, "TTFT (s)": 0, "Tokens": 0, "Tokens/sec": 0}

def run_synthetic_cpu_model(script_name):
    # Runs either bad_model.py or optimized_model.py
    model_id = script_name.replace(".py", "")
    try:
        start_time = time.time()
        result = subprocess.run(["python", script_name], capture_output=True, text=True)
        end_time = time.time()
        
        output = result.stdout
        tokens = 0
        tps = 0.0
        try:
            for line in output.split('\n'):
                if line.startswith("Generated ") and "tokens" in line:
                    tokens = int(line.split(" ")[1])
                if line.startswith("Throughput:"):
                    span = line.split("Throughput: ")[1]
                    tps = float(span.split(" ")[0])
        except Exception:
            pass
            
        total_time = end_time - start_time
        return {
            "Model": model_id,
            "Status": "✅ Success",
            "Total Time (s)": round(total_time, 2),
            "TTFT (s)": round(total_time * 0.2, 2) if tokens > 0 else 0, # Approximation for synthetic
            "Tokens": tokens,
            "Tokens/sec": round(tps, 2)
        }
    except Exception as e:
        return {"Model": model_id, "Status": f"❌ Error: {e}", "Total Time (s)": 0, "TTFT (s)": 0, "Tokens": 0, "Tokens/sec": 0}

# --- Main Application Logic ---

if st.sidebar.button("🚀 Run Profiler & Optimizer"):
    if not selected_models:
        st.warning("Please select at least one model to run.")
        st.stop()

    st.markdown("### 📡 Live Resource Telemetry")
    
    # Placeholders for top-level metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    cpu_metric = col1.empty()
    gpu_metric = col2.empty()
    vram_bw_metric = col3.empty()
    power_metric = col4.empty()
    io_metric = col5.empty()

    chart_placeholder = st.empty()
    status_placeholder = st.empty()
    
    tracking_data = {
        "Time (s)": [],
        "CPU (%)": [],
        "GPU (%)": [],
        "VRAM Bandwidth (%)": [],
        "Power (W)": [],
        "Disk I/O (MB/s)": []
    }
    
    # Execution Engine
    def execute_models():
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(selected_models))) as executor:
            future_to_model = {}
            for m in selected_models:
                if m == "bad_model":
                    future_to_model[executor.submit(run_synthetic_cpu_model, "bad_model.py")] = m
                elif m == "optimized_model":
                    future_to_model[executor.submit(run_synthetic_cpu_model, "optimized_model.py")] = m
                else:
                    future_to_model[executor.submit(run_ollama_model, m, prompt_text)] = m
            
            for future in concurrent.futures.as_completed(future_to_model):
                results.append(future.result())
        return results

    # Launch background thread
    start_time = time.time()
    future_results = None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as main_executor:
        future_results = main_executor.submit(execute_models)
        
        last_io = psutil.disk_io_counters()
        last_io_time = time.time()
        
        while not future_results.done():
            current_time = time.time()
            elapsed_sys = current_time - start_time
            
            cpu_pct = psutil.cpu_percent(interval=None)
            
            io_now = psutil.disk_io_counters()
            io_bytes = (io_now.read_bytes + io_now.write_bytes) - (last_io.read_bytes + last_io.write_bytes)
            io_mbs = (io_bytes / (current_time - last_io_time)) / (1024*1024) if current_time > last_io_time else 0
            
            last_io = io_now
            last_io_time = current_time
            
            if has_gpu:
                util = nvmlDeviceGetUtilizationRates(handle)
                gpu_pct = util.gpu
                vram_bw_pct = util.memory # Represents Memory Bandwidth Utilization
                try:
                    power_w = nvmlDeviceGetPowerUsage(handle) / 1000.0
                except NVMLError:
                    power_w = 0.0
            else:
                gpu_pct, vram_bw_pct, power_w = 0, 0, 0
                
            tracking_data["Time (s)"].append(elapsed_sys)
            tracking_data["CPU (%)"].append(cpu_pct)
            tracking_data["GPU (%)"].append(gpu_pct)
            tracking_data["VRAM Bandwidth (%)"].append(vram_bw_pct)
            tracking_data["Power (W)"].append(power_w)
            tracking_data["Disk I/O (MB/s)"].append(io_mbs)
            
            cpu_metric.metric("Avg CPU Usage", f"{cpu_pct:.1f} %")
            gpu_metric.metric("GPU Compute Util", f"{gpu_pct:.1f} %")
            vram_bw_metric.metric("VRAM Bandwidth", f"{vram_bw_pct:.1f} %")
            power_metric.metric("GPU Power Draw", f"{power_w:.1f} W")
            io_metric.metric("Disk I/O", f"{io_mbs:.2f} MB/s")
            
            df_chart = pd.DataFrame(tracking_data).set_index("Time (s)")
            chart_placeholder.line_chart(df_chart)
            
            status_placeholder.info(f"⏳ **Models are running concurrently...** Elapsed: {elapsed_sys:.1f}s")
            time.sleep(0.5)
            
    # Process Results
    final_results = future_results.result()
    total_time_taken = time.time() - start_time
    status_placeholder.success(f"✅ **Benchmark Complete!** Total Wall-Clock Time: {total_time_taken:.2f}s")

    st.markdown("---")
    
    # --- Optimization Proof Section ---
    df_results = pd.DataFrame(final_results)
    
    did_run_comparison = "bad_model" in df_results['Model'].values and "optimized_model" in df_results['Model'].values
    if did_run_comparison:
        st.subheader("🏆 Optimization Proof: Vectorization & KV Caching")
        st.markdown("We specifically deployed **`optimized_model.py`** alongside **`bad_model.py`** to demonstrate a targeted improvement to the CPU Bottleneck.")
        
        comp_df = df_results[df_results['Model'].isin(["bad_model", "optimized_model"])]
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Throughput Comparison (Tokens/sec) 📈**")
            st.bar_chart(comp_df.set_index("Model")["Tokens/sec"], color="#00ffcc")
        with c2:
            st.markdown("**Execution Time Drop (Seconds) 📉**")
            st.bar_chart(comp_df.set_index("Model")["Total Time (s)"], color="#ff4b4b")
            
        bad_tps = float(comp_df[comp_df["Model"] == "bad_model"]["Tokens/sec"].iloc[0])
        opt_tps = float(comp_df[comp_df["Model"] == "optimized_model"]["Tokens/sec"].iloc[0])
        
        if bad_tps > 0:
            speedup = opt_tps / bad_tps
            st.success(f"**Optimization Result**: By replacing O(N³) Python loops with **NumPy C-Contiguous BLAS Vectorization** and implementing a computational **KV Cache**, the model achieved a **{speedup:.1f}x speedup** in identical throughput workload!")

    st.markdown("---")
    st.subheader("📊 Output Metrics")
    
    st.dataframe(df_results, use_container_width=True)
    
    # AI Insight Engine
    st.markdown("---")
    st.subheader("🤖 Automated Diagnostic Insight (AI Analysis)")
    
    available_ai = [m for m in available_models if m in ["llama3", "mistral", "phi3"] and m in selected_models]
    if not available_ai:
         available_ai = [m for m in available_models if m in ["llama3", "mistral", "phi3"]]
    
    if available_ai:
        analyst_model = available_ai[0]
        with st.spinner(f"Using **{analyst_model}** to analyze performance bottlenecks..."):
            max_metrics = {
                "Max CPU": max(tracking_data["CPU (%)"], default=0),
                "Max GPU": max(tracking_data["GPU (%)"], default=0),
                "Max VRAM Bandwidth": max(tracking_data["VRAM Bandwidth (%)"], default=0),
                "Max Power": max(tracking_data["Power (W)"], default=0),
                "Max Disk I/O": max(tracking_data["Disk I/O (MB/s)"], default=0)
            }
            
            ai_prompt = f"""
            You are a system performance engineer. Analyze this data from a hardware benchmark.
            Models run concurrently: {', '.join(selected_models)}.
            Total Time: {total_time_taken:.2f}s.
            
            Max Hardware Spikes During Execution:
            - CPU: {max_metrics['Max CPU']:.1f}%
            - GPU: {max_metrics['Max GPU']:.1f}%
            - VRAM Bandwidth: {max_metrics['Max VRAM Bandwidth']:.2f} %
            - Power: {max_metrics['Max Power']:.1f} W
            - Disk I/O: {max_metrics['Max Disk I/O']:.2f} MB/s
            
            Model execution table:
            {df_results.to_string(index=False)}
            
            Provide a highly structured analysis. Use EXACTLY this markdown format, with bullet points:
            
            ### 🚨 Identified Bottlenecks
            * **[Component]**: [Brief description of what reached its limit]
            
            ### 🔍 Root Causes
            * **[Model/Process Name]**: [Why the model caused this issue. If bad_model was run, state its CPU bloat. If optimized_model was run, commend the vectorization]
            
            ### 🛠️ Recommended Optimizations
            * **[Action]**: [Clear, actionable step]
            
            Do not include any paragraph explanations or conversational text. Stop after the three sections.
            """
            
            try:
                response = ollama.chat(model=analyst_model, messages=[{"role": "user", "content": ai_prompt}])
                st.markdown(response['message']['content'])
            except Exception as e:
                st.error(f"Insight Generation Failed: Make sure {analyst_model} is pulled in Ollama. Error: {e}")
    else:

        st.warning("No LLM available to generate insights (please install llama3, mistral, or phi3).")
