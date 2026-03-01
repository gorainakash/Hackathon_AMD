# 🧩 ConcurML  
### ⚡ Performance Intelligence for Concurrent AI Systems  

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-red.svg" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />
  <img src="https://img.shields.io/badge/Status-Active-success.svg" />
  <img src="https://img.shields.io/badge/AI-LLM%20Powered-purple.svg" />
</p>

<p align="center">
  <b>Monitor. Diagnose. Optimize.</b><br/>
  Built for modern AI workloads running at scale.
</p>


---

## 🚀 What is ConcurML?

ConcurML is a real-time performance intelligence platform designed to profile and optimize **concurrent AI/ML workloads**.

It bridges the gap between:
AI Model Execution → Hardware Utilization → Bottleneck Detection → Optimization Proof


Unlike traditional system monitors, ConcurML doesn’t just show numbers —  
it explains performance behavior and validates measurable improvements.

---

### 🔧 Requirements

- Python 3.10+
- Ollama installed locally (with llama3)
- NVIDIA GPU (optional, for GPU telemetry)
- Streamlit

---

## ⚠️ The Challenge

Modern AI deployments face:

- CPU saturation during concurrent inference  
- GPU underutilization or memory overflow  
- Unpredictable latency under load  
- Lack of optimization validation  

Existing tools show metrics.  
They don’t connect them to AI execution logic.

---

## 💡 The ConcurML Approach

✔ Run multiple models concurrently  
✔ Capture live hardware telemetry  
✔ Detect bottlenecks automatically  
✔ Benchmark optimized vs naive execution  
✔ Generate structured AI-driven diagnostics  

From raw telemetry to actionable intelligence.

---


## 🔴 bad_model.py

# The naive implementation:
-Uses nested Python loops
-Runs purely in interpreted Python
-Creates high interpreter overhead
-Fails to leverage BLAS or SIMD optimizations
-Causes CPU core saturation

# Impact:
-High execution time
-Low throughput (tokens/sec)
-CPU bottleneck due to software inefficiency

## 🟢 optimized_model.py

# The optimized implementation:
-Uses NumPy vectorization (C-backed BLAS execution)
-Reduces Python-level looping
-Improves memory locality and cache usage
-Utilizes multi-core CPU instructions efficiently

# Impact:
-Significantly reduced execution time
-Higher throughput
-Lower CPU saturation
-Improved scalability under concurrency

---

## ✨ Key Capabilities

- 🚀 Parallel multi-model execution engine  
- 📡 Real-time CPU, GPU, VRAM & I/O tracking  
- 🛡 Predictive resource estimation  
- 📊 Interactive performance dashboard  
- 🏆 Optimization benchmarking framework  
- 🤖 Automated bottleneck classification using LLMs  

---

## 🏗 System Architecture


User Interface (Streamlit)
↓
Concurrent Execution Engine
↓
Model Layer (LLMs + Synthetic Models)
↓
Telemetry Layer (CPU / GPU Monitoring)
↓
Analytics Engine
↓
AI Diagnostic Report


---

## 🛠 Technology Stack

| Layer | Technology |
|-------|------------|
| Compute | AMD Ryzen Processor |
| Backend | Python |
| UI | Streamlit |
| AI Engine | Ollama (LLMs) |
| Telemetry | psutil + NVIDIA NVML |
| Analytics | Pandas |

---

## ⚡ Quick Start

```bash
git clone https://github.com/gorainakash/ConcurML.git
cd Hackathon_AMD
pip install -r requirements.txt
streamlit run MAIN.py
