<div style="font-family:Arial, sans-serif; line-height:1.6; max-width:900px; margin:auto; padding:20px;">

  <p align="center">
    <img src="https://github.com/microsoft/onnxruntime/raw/main/docs/images/ONNX_Runtime_logo_dark.png" alt="ONNX Runtime Logo" width="320"/>
  </p>

  <h1>Welcome to the ONNX Runtime – Execution Provider Coverage Tester</h1>

  <p>
    This open source initiative, led by <strong><a href="https://graiphic.io/" target="_blank">Graiphic</a></strong>, provides 
    a detailed, real-world coverage map of ONNX operator support for each <strong>Execution Provider (EP)</strong> in 
    <strong><a href="https://github.com/microsoft/onnxruntime" target="_blank">ONNX Runtime</a></strong>.
  </p>

  <p>
    It is part of our broader effort to democratize AI deployment through 
    <a href="https://graiphic.io/download/" target="_blank"><strong>SOTA</strong></a> — 
    an ONNX-native orchestration framework designed for engineers, researchers, and industrial use cases.
  </p>

  <h2>🎯 Project Objectives</h2>
  <ul>
    <li>Systematically test and report ONNX operator coverage per Execution Provider.</li>
    <li>Deliver up-to-date insights to guide industrial and academic ONNX Runtime adoption.</li>
    <li>Help developers, maintainers, and hardware vendors prioritize missing or broken operator support.</li>
  </ul>

  <h2>🧪 What’s Tested</h2>
  <ul>
    <li>Each ONNX operator is tested in isolation across all supported EPs.</li>
    <li>Results include status per operator: <code>SUCCESS</code>, <code>FALLBACK</code>, <code>FAIL</code>, <code>NOT TESTED</code>, <code>SKIPPED</code>, <code>UNKNOWN</code>.</li>
    <li>Each EP includes a complete dataset with test logs, node-level breakdowns, and global stats.</li>
  </ul>

  <h2>📦 Currently Supported Execution Providers</h2>
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
    While the official ONNX Runtime documentation is robust, it does not provide a detailed and up-to-date mapping 
    of operator support per Execution Provider. In reality, compatibility often varies based on:
  </p>

  <ul>
    <li>Runtime version and hardware driver versions (e.g., CUDA, cuDNN, OpenVINO)</li>
    <li>Execution Provider implementation status and fallback logic</li>
    <li>Graph structure and data type constraints</li>
  </ul>

  <p>
    <strong>This project fills that gap.</strong> By testing every ONNX operator across each EP, we provide:
  </p>

  <ul>
    <li>A precise <strong>operator coverage map</strong> to help developers identify regressions or missing features</li>
    <li>A <strong>real-time view of deployment readiness</strong> to help users choose the right hardware and provider</li>
    <li>A <strong>trust layer</strong> that accelerates ONNX Runtime adoption in production and research environments</li>
  </ul>

  <p>
    Ultimately, this initiative supports the global ONNX community by bringing clarity and visibility to the Runtime layer — 
    essential for both reliability and performance in real-world use.
  </p>

  <h2>🚀 How to Use</h2>
  <ul>
    <li>Navigate to each EP folder to explore environment details and results.</li>
    <li>Use global statistics and charts to compare EP maturity.</li>
    <li>Contribute by testing on your own platform and submitting new data!</li>
  </ul>

  <h2>📊 Summary of ONNX Execution Provider Results</h2>

  <p>This summary reflects the latest test run on the following configuration:</p>

  <ul>
    <li><strong>CPU:</strong> Intel(R) Core(TM) i7-9700 @ 3.00GHz</li>
    <li><strong>GPU:</strong> NVIDIA GeForce RTX 2070</li>
    <li><strong>CUDA:</strong> 12.5 | <strong>cuDNN:</strong> 9.2.1 | <strong>TensorRT:</strong> 10.9.0</li>
    <li><strong>ONNX:</strong> 1.18.0 | <strong>ONNXRuntime:</strong> 1.23.0</li>
    <li><strong>OS:</strong> Windows 10</li>
  </ul>

  <table border="1" cellpadding="6" cellspacing="0">
    <thead>
      <tr>
        <th>Execution Provider</th>
        <th>SUCCESS</th>
        <th>FALLBACK</th>
        <th>FAIL</th>
        <th>NOT TESTED</th>
        <th>SKIPPED</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><a href="https://github.com/Graiphic/ONNX-Runtime/tree/main/Execution%20Providers%20Tester/CPU" target="_blank">CPU</a></td>
        <td>211 (80%)</td><td>0 (0%)</td><td>45 (17%)</td><td>7 (3%)</td><td>0 (0%)</td>
      </tr>
      <tr>
        <td><a href="https://github.com/Graiphic/ONNX-Runtime/tree/main/Execution%20Providers%20Tester/Intel%20-%20oneDNN" target="_blank">Intel – oneDNN</a></td>
        <td>45 (17%)</td><td>165 (63%)</td><td>46 (17%)</td><td>7 (3%)</td><td>0 (0%)</td>
      </tr>
      <tr>
        <td><a href="https://github.com/Graiphic/ONNX-Runtime/tree/main/Execution%20Providers%20Tester/Intel%20-%20OpenVINO%E2%84%A2" target="_blank">Intel – OpenVINO™</a></td>
        <td>74 (28%)</td><td>132 (50%)</td><td>50 (19%)</td><td>7 (3%)</td><td>0 (0%)</td>
      </tr>
      <tr>
        <td><a href="https://github.com/Graiphic/ONNX-Runtime/tree/main/Execution%20Providers%20Tester/Nvidia%20-%20CUDA" target="_blank">NVIDIA – CUDA</a></td>
        <td>127 (48%)</td><td>112 (43%)</td><td>17 (6%)</td><td>7 (3%)</td><td>0 (0%)</td>
      </tr>
      <tr>
        <td><a href="https://github.com/Graiphic/ONNX-Runtime/tree/main/Execution%20Providers%20Tester/Nvidia%20-%20TensorRT" target="_blank">NVIDIA – TensorRT</a></td>
        <td>94 (36%)</td><td>139 (53%)</td><td>23 (9%)</td><td>7 (3%)</td><td>0 (0%)</td>
      </tr>
      <tr>
        <td><a href="https://github.com/Graiphic/ONNX-Runtime/tree/main/Execution%20Providers%20Tester/Windows%20-%20DirectML" target="_blank">Windows – DirectML</a></td>
        <td>127 (48%)</td><td>84 (32%)</td><td>45 (17%)</td><td>7 (3%)</td><td>0 (0%)</td>
      </tr>
    </tbody>
  </table>

  <h2>🤝 Maintainer</h2>
  <p>
    This project is maintained by <strong><a href="https://graiphic.io/" target="_blank">Graiphic</a></strong> 
    as part of the <a href="https://graiphic.io/download/" target="_blank"><strong>SOTA</strong></a> initiative.
  </p>
  <p>
    We welcome collaboration, community feedback, and open contribution to make ONNX Runtime stronger and more widely adopted.
  </p>

  <p style="margin-top:20px;">
    📬 <strong>Contact:</strong> <a href="mailto:contact@graiphic.io">contact@graiphic.io</a><br>
    🌐 <strong>Website:</strong> <a href="https://graiphic.io/" target="_blank">graiphic.io</a><br>
    🧠 <strong>Learn more about SOTA:</strong> <a href="https://graiphic.io/download/" target="_blank">graiphic.io/download</a>
  </p>

</div>
