# Graiphic – ONNX / ONNX Runtime Contributions

Welcome to the official ONNX Runtime contribution repository maintained by **Graiphic**.

We are a French deep-tech company building tools to make AI and graph computing truly accessible and transparent across industries. Our core vision is to deliver a **universal AI orchestration framework based on ONNX and ONNX Runtime**, fully integrated with **LabVIEW** to bring AI workflows into test, measurement, research, and embedded environments.

---

## 🌐 Our Mission with ONNX Runtime

Graiphic is committed to contributing to the ONNX and ONNX Runtime ecosystem through a series of open source, industry-driven, and pragmatic initiatives.

Our goals:
- 🧩 **Accessibility** – Make ONNX Runtime easy to use, even for non-programmers, via a visual environment (LabVIEW).
- 🔌 **Interoperability** – Enable seamless integration across hardware targets (CPU, GPU, FPGA, NPU).
- 🧱 **Modularity** – Allow full graph orchestration, modification, and control without leaving the ONNX domain.
- ⚡ **Efficiency** – Optimize deployment for real-world, latency-sensitive, or resource-constrained environments.

These principles are embodied in our flagship platform, [**SOTA**](https://graiphic.io/download/): the first visual and fully ONNX-native AI development suite.

---

## 🔎 Project 1 – Execution Providers Coverage Tester

📍 Repository: [Execution Providers Tester](https://github.com/Graiphic/ONNX-Runtime/tree/main/Execution%20Providers%20Tester)

This project provides a **systematic and open evaluation of ONNX Runtime operator coverage** for each Execution Provider (EP).  
It aims to answer two recurring questions from industrial partners:

1. ✅ What is the exact set of ONNX operators supported natively by each EP?
2. 📉 How complete is the implementation versus what falls back or fails?

### Key Features:
- Full ONNX node-by-node execution testing
- EPs currently supported: CPU, CUDA, TensorRT, OpenVINO, oneDNN, DirectML
- Real-world statistics with SUCCESS / FALLBACK / FAIL / NOT TESTED mapping
- Individual reports per EP including environment details

👉 This tool is both:
- A **diagnostic tool** for developers and maintainers
- A **decision-making aid** for industrial users selecting ONNX Runtime for deployment

Coming soon: support for AMD ROCm and Vitis AI.

---

## 🚧 Upcoming Graiphic Contributions

We're currently working on other ONNX Runtime-related initiatives, including:

- 📊 **Benchmark Toolkit** – to compare ONNX Runtime execution vs native implementations (e.g., OpenCV, PyTorch) on various targets and hardware setups
- 🧠 **Training graph extension** – extending ONNX Runtime toward supervised and reinforcement training orchestration (not just inference)
- 🕸️ **Visual AI composer** – for designing, editing, and deploying ONNX graphs directly from a LabVIEW-native interface
- 📡 **Real-time synchronization tools** – to embed ONNX graphs into hardware-driven acquisition, timestamping, and test benches

---

## 💡 Why It Matters

The ONNX Runtime documentation is powerful but limited in terms of real-time, granular coverage of EP capabilities.  
Graiphic provides an essential **complementary layer** that:

- Improves **transparency** for developers and vendors
- Accelerates **industrial adoption**
- Enhances **trust and control** for graph-based AI deployment in regulated or critical environments

---

## 👥 About Graiphic

We are building the future of industrial AI orchestration — bridging deep learning, system engineering, and embedded hardware.  
Our flagship product [SOTA](https://graiphic.io/download/) is the first ONNX-native framework to make training, inference, and graph manipulation fully visual and modular.

🔗 Website: [https://graiphic.io](https://graiphic.io)  
📬 Contact: contact@graiphic.io

---

Join us in shaping the future of ONNX-powered AI workflows 🚀
