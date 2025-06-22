<div style="font-family:Arial, sans-serif; line-height:1.6; max-width:800px; margin:auto; padding:20px;">

  <h1>Welcome to the ONNX Runtime – Execution Provider Coverage Tester</h1>

  <p>
    This open source project, initiated by <strong><a href="https://graiphic.io/" target="_blank">Graiphic</a></strong>,
    provides a detailed, real-world map of operator support per <strong>Execution Provider (EP)</strong> in 
    <strong>ONNX Runtime</strong>.
  </p>

  <p>
    It is part of our broader effort to democratize AI deployment through
    <a href="https://graiphic.io/download/" target="_blank"><strong>SOTA</strong></a> — 
    a native ONNX orchestration framework designed for engineers, researchers, and industrial users.
  </p>

  <h2>🎯 Project Objectives</h2>
  <ul>
    <li>Systematically test and report operator coverage per Execution Provider.</li>
    <li>Provide clear, up-to-date visibility on ONNX Runtime support for hardware adoption decisions.</li>
    <li>Help developers and integrators identify missing or failing operator support.</li>
  </ul>

  <h2>🧪 What’s Tested</h2>
  <ul>
    <li>Each standard ONNX operator is executed in isolation on each EP.</li>
    <li>Status per operator: <code>SUCCESS</code>, <code>FALLBACK</code>, <code>FAIL</code>, <code>NOT TESTED</code>, <code>SKIPPED</code>, <code>UNKNOWN</code>.</li>
    <li>Detailed reports and visualizations are generated for each EP.</li>
  </ul>

  <h2>📦 Current Execution Providers</h2>
  <ul>
    <li>CPU</li>
    <li>NVIDIA – CUDA</li>
    <li>NVIDIA – TensorRT</li>
    <li>Intel – OpenVINO</li>
    <li>Intel – oneDNN</li>
    <li>Windows – DirectML</li>
    <li><em>Coming soon:</em> AMD – Vitis AI and ROCm</li>
  </ul>

  <h2>📍 Why This Project Matters</h2>
  <p>
    The official ONNX Runtime documentation provides essential information, but lacks detailed and real-time visibility 
    into which ONNX operators are actually supported per execution provider.
  </p>
  <p>
    In real-world usage, behavior varies depending on:
  </p>
  <ul>
    <li>ONNX Runtime and driver/library versions (e.g., CUDA, DNN, OpenVINO)</li>
    <li>Model structure and graph-specific dependencies</li>
    <li>Hardware execution path and fallback logic</li>
  </ul>

  <p>
    <strong>This project fills that gap.</strong> It complements ONNX Runtime by systematically testing and publishing 
    the full operator coverage of each EP. This provides:
  </p>

  <ul>
    <li>
      A <strong>detailed operator coverage map</strong> to help developers detect regressions, bugs,
      or unimplemented operations.
    </li>
    <li>
      A <strongreal-time reference</strong> for users evaluating ONNX Runtime deployment across CPUs, GPUs, NPUs, and FPGAs.
    </li>
    <li>
      A strong contribution to <strong>industrial and research adoption</strong> by improving transparency,
      compatibility checks, and confidence in ONNX Runtime as a production-grade stack.
    </li>
  </ul>

  <p>
    Our goal is to make ONNX Runtime easier to trust, adopt, and integrate —
    across all sectors of engineering, AI, and testing infrastructures.
  </p>

  <h2>🚀 How to Use</h2>
  <p>
    Each EP folder includes:
  </p>
  <ul>
    <li>Test environment description</li>
    <li>Operator-by-operator results</li>
    <li>Global summary charts</li>
    <li>Detailed statistics (success, fallback, failure rates)</li>
  </ul>

  <p>
    You can use this to select the best Execution Provider for your model and hardware setup.
  </p>
</div>


# Summary of ONNX Execution Provider Results

This document gathers all test results by Execution Provider (EP).
Each EP has its own `README.md` with detailed results.

## Hardware and Software

- **CPU:** Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz
- **GPU(s):** NVIDIA GeForce RTX 2070
- **CUDA version:** 12.5
- **cuDNN version:** 9.2.1
- **TensorRT version:** 10.9.0
- **ONNX version:** 1.18.0
- **ONNXRuntime version:** 1.23.0
- **Operating System (OS):** Windows 10

## Summary Table

| Execution Provider | SUCCESS | FALLBACK | FAIL | NOT TESTED | SKIPPED |
|:------------------:|:-------:|:--------:|:----:|:-----------:|:--------:|
| CPU | 211 (80%) | 0 (0%) | 45 (17%) | 7 (3%) | 0 (0%) |
| Intel - oneDNN | 45 (17%) | 165 (63%) | 46 (17%) | 7 (3%) | 0 (0%) |
| Intel - OpenVINO™ | 74 (28%) | 132 (50%) | 50 (19%) | 7 (3%) | 0 (0%) |
| NVIDIA - CUDA | 127 (48%) | 112 (43%) | 17 (6%) | 7 (3%) | 0 (0%) |
| NVIDIA - TensorRT | 94 (36%) | 139 (53%) | 23 (9%) | 7 (3%) | 0 (0%) |
| Windows - DirectML | 127 (48%) | 84 (32%) | 45 (17%) | 7 (3%) | 0 (0%) |


<h2>🤝 Maintainer</h2>
  <p>
    This project is maintained by <strong><a href="https://graiphic.io/" target="_blank">Graiphic</a></strong> as part of 
    the <a href="https://graiphic.io/download/" target="_blank"><strong>SOTA</strong></a> initiative.
    We welcome collaborations, issue reports, and community contributions.
  </p>

  <p style="margin-top:20px;">
    📬 <strong>Contact:</strong> <a href="mailto:contact@graiphic.io">contact@graiphic.io</a> <br>
    🌐 <strong>Website:</strong> <a href="https://www.graiphic.io" target="_blank">graiphic.io</a>
  </p>
