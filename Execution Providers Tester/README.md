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

<h2>📐 How’s Tested</h2>
<p>
  Each operator is tested using a <strong>minimal ONNX model</strong> containing only the node under test and its required inputs.
  These models are constructed dynamically for each operator, and executed with the target <strong>Execution Provider (EP)</strong>.
</p>
<p>
  In most cases, this provides a direct and unambiguous signal of EP support: if a node runs successfully in isolation, it is considered
  <strong>natively supported</strong>.
</p>
<p>
  However, for some EPs such as <strong>OpenVINO</strong> and <strong>TensorRT</strong>, fallback to CPU may occur even if the node is technically supported.
  This can be due to backend heuristics requiring a minimal graph complexity to activate EP-specific execution paths.
  In such cases, we attempt a <strong>model complexification step</strong> by embedding the node in a richer context (e.g., with dummy operations).
  If the node then executes on the EP, its status is upgraded to <code>SUCCESS (with complexification)</code>.
</p>


<h2>📦 Currently Supported Execution Providers</h2>
<ul>
<li><a href="./CPU/" target="_blank">CPU</a></li>
<li><a href="./Intel%20-%20OneDNN/" target="_blank">Intel - OneDNN</a></li>
<li><a href="./Intel%20-%20OpenVINO%E2%84%A2/" target="_blank">Intel - OpenVINO™</a></li>
<li><a href="./Nvidia%20-%20CUDA/" target="_blank">Nvidia - CUDA</a></li>
<li><a href="./Nvidia%20-%20TensorRT/" target="_blank">Nvidia - TensorRT</a></li>
<li><a href="./Windows%20-%20DirectML/" target="_blank">Windows - DirectML</a></li>
<li><em>Coming soon:</em> AMD – Vitis AI and ROCm</li>
</ul>

<h2>📊 Summary of ONNX Execution Provider Results</h2>
<p>This summary reflects the latest test run on the following configuration:</p>
<ul>
  <li><strong>CPU:</strong> Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz</li>
  <li><strong>GPU:</strong> NVIDIA GeForce RTX 2070</li>
  <li><strong>CUDA:</strong> 12.5 | <strong>cuDNN:</strong> 9.2.1 | <strong>TensorRT:</strong> 10.9.0</li>
  <li><strong>ONNX:</strong> 1.18.0 | <strong>ONNXRuntime:</strong> 1.23.0</li>
  <li><strong>OS:</strong> Windows 10</li>
</ul>

<h3>⚙️ Test Configuration</h3>
<ul>
  <li><strong>ONNX Opset version:</strong> 22</li>
  <li><strong>ONNX IR version:</strong> 10</li>
  <li><strong>Data types:</strong> Only a single data type is tested per node. In general this is <code>float32</code>,
      unless the node does not support it—in which case an available type is selected.</li>
</ul>
<p><strong>Note:</strong> Some ONNX nodes may not be available on the selected Execution Provider (EP) for opset version 22. 
This can lead to fallback behavior even though these nodes were supported in earlier opset versions. 
This occurs because ONNX Runtime teams may not have implemented or updated certain operators for the latest opset. 
As a result, test outcomes can vary depending on both the ONNX opset version and the ONNX Runtime version used.</p>
<h3>ONNX Core Operators</h3>
<table border="1" cellpadding="6" cellspacing="0">
  <thead>
    <tr>
      <th>Execution Provider</th>
      <th>SUCCESS</th>
      <th>FALLBACK</th>
      <th>SUPPORTED</th>
      <th>FAIL</th>
      <th>NOT TESTED</th>
      <th>SKIPPED</th>
    </tr>
  </thead>
  <tbody>
<tr><td><a href="./CPU/" target="_blank">CPU</a></td><td>142 (93%)</td><td>0 (0%)</td><td>142 (93%)</td><td>11 (7%)</td><td>0 (0%)</td><td>0 (0%)</td></tr>
<tr><td><a href="./Intel%20-%20OneDNN/" target="_blank">Intel - OneDNN</a></td><td>39 (25%)</td><td>103 (67%)</td><td>142 (93%)</td><td>11 (7%)</td><td>0 (0%)</td><td>0 (0%)</td></tr>
<tr><td><a href="./Intel%20-%20OpenVINO%E2%84%A2/" target="_blank">Intel - OpenVINO™</a></td><td>93 (61%)</td><td>48 (31%)</td><td>141 (92%)</td><td>12 (8%)</td><td>0 (0%)</td><td>0 (0%)</td></tr>
<tr><td><a href="./Nvidia%20-%20CUDA/" target="_blank">Nvidia - CUDA</a></td><td>70 (46%)</td><td>72 (47%)</td><td>142 (93%)</td><td>11 (7%)</td><td>0 (0%)</td><td>0 (0%)</td></tr>
<tr><td><a href="./Nvidia%20-%20TensorRT/" target="_blank">Nvidia - TensorRT</a></td><td>88 (58%)</td><td>57 (37%)</td><td>145 (95%)</td><td>8 (5%)</td><td>0 (0%)</td><td>0 (0%)</td></tr>
<tr><td><a href="./Windows%20-%20DirectML/" target="_blank">Windows - DirectML</a></td><td>97 (63%)</td><td>44 (29%)</td><td>141 (92%)</td><td>12 (8%)</td><td>0 (0%)</td><td>0 (0%)</td></tr>
</tbody></table>

<p>
  The ONNX Core Operators table reflects the support status of standard <strong>ONNX specification operators</strong>. 
  These are the official operators defined by the ONNX community, and represent the majority of common model operations.
</p>
<h3>Microsoft Custom Operators</h3>
<table border="1" cellpadding="6" cellspacing="0">
  <thead>
    <tr>
      <th>Execution Provider</th>
      <th>SUCCESS</th>
      <th>FALLBACK</th>
      <th>SUPPORTED</th>
      <th>FAIL</th>
      <th>NOT TESTED</th>
      <th>SKIPPED</th>
    </tr>
  </thead>
  <tbody>
<tr><td><a href="./CPU/" target="_blank">CPU</a></td><td>59 (55%)</td><td>0 (0%)</td><td>59 (55%)</td><td>41 (38%)</td><td>7 (7%)</td><td>0 (0%)</td></tr>
<tr><td><a href="./Intel%20-%20OneDNN/" target="_blank">Intel - OneDNN</a></td><td>6 (6%)</td><td>52 (49%)</td><td>58 (54%)</td><td>42 (39%)</td><td>7 (7%)</td><td>0 (0%)</td></tr>
<tr><td><a href="./Intel%20-%20OpenVINO%E2%84%A2/" target="_blank">Intel - OpenVINO™</a></td><td>14 (13%)</td><td>42 (39%)</td><td>56 (52%)</td><td>44 (41%)</td><td>7 (7%)</td><td>0 (0%)</td></tr>
<tr><td><a href="./Nvidia%20-%20CUDA/" target="_blank">Nvidia - CUDA</a></td><td>53 (50%)</td><td>34 (32%)</td><td>87 (81%)</td><td>13 (12%)</td><td>7 (7%)</td><td>0 (0%)</td></tr>
<tr><td><a href="./Nvidia%20-%20TensorRT/" target="_blank">Nvidia - TensorRT</a></td><td>6 (6%)</td><td>79 (74%)</td><td>85 (79%)</td><td>15 (14%)</td><td>7 (7%)</td><td>0 (0%)</td></tr>
<tr><td><a href="./Windows%20-%20DirectML/" target="_blank">Windows - DirectML</a></td><td>27 (25%)</td><td>33 (31%)</td><td>60 (56%)</td><td>40 (37%)</td><td>7 (7%)</td><td>0 (0%)</td></tr>
</tbody></table>

<p>
  The Microsoft Custom Operators table lists <strong>proprietary or domain-specific extensions</strong> provided by ONNX Runtime, 
  typically used in models exported from tools like Olive, Azure ML, or other Microsoft pipelines.
</p>

<h4>Legend:</h4>
<ul>
  <li><strong>SUCCESS</strong>: Node executed natively by the Execution Provider.</li>
  <li><strong>FALLBACK</strong>: Node executed by a fallback mechanism (typically CPU).</li>
  <li><strong>SUPPORTED</strong>: Sum of SUCCESS and FALLBACK, indicates total operability.</li>
  <li><strong>FAIL</strong>: Node failed execution even with fallback enabled.</li>
  <li><strong>NOT TESTED</strong>: Node was not tested for this provider (unsupported type or config).</li>
  <li><strong>SKIPPED</strong>: Node was deliberately skipped due to known incompatibility or user choice.</li>
</ul>
<h2>🧭 Related Tools</h2>
<p>
  For a complementary and more aggregated perspective on backend compliance,
  we encourage you to also visit the official 
  <a href="https://onnx.ai/backend-scoreboard/" target="_blank"><strong>ONNX Backend Scoreboard</strong></a>.
</p>
<p>
  While the Scoreboard provides a high-level view of backend support based on ONNX's internal test suite,
  our initiative focuses on operator-level validation and runtime behavior analysis — especially fallback detection — across Execution Providers.
  Together, both efforts help build a clearer, more actionable picture of ONNX Runtime capabilities.
</p>
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
