<div style="font-family:Arial, sans-serif; line-height:1.6; max-width:900px; margin:auto; padding:20px;">

<p align="center">
  <img src="https://github.com/microsoft/onnxruntime/raw/main/docs/images/ONNX_Runtime_logo_dark.png" alt="ONNX Runtime Logo" width="320"/>
</p>

<h1>ONNX Runtime — EP Coverage (Opset 20)</h1>

<p>
  This open source initiative, led by <strong><a href="https://graiphic.io/" target="_blank">Graiphic</a></strong>, provides 
  a detailed, real-world coverage map of ONNX operator support for each <strong>Execution Provider (EP)</strong> in 
  <strong><a href="https://github.com/microsoft/onnxruntime" target="_blank">ONNX Runtime</a></strong>.
</p>

<h2>🧪 What’s Tested</h2>
<ul>
  <li>Each ONNX operator is tested in isolation using a minimal single-node model.</li>
  <li>Status per operator: <code>SUCCESS</code>, <code>FALLBACK</code>, <code>FAIL</code>, <code>NOT TESTED</code>, <code>SKIPPED</code>, <code>UNKNOWN</code>.</li>
  <li>Per-EP datasets include logs, optimized models (when applicable), and a README.</li>
</ul>

<h2>📐 How’s Tested</h2>
<h3>Inference</h3>
<p>
  Each operator is tested with a minimal ONNX graph. For EPs like OpenVINO/TensorRT, a <em>complexification</em> pass can add a small chain
  of <code>Mul</code>/<code>And</code> nodes (type-dependent) to make the backend compile more of the graph and reveal actual EP coverage.
</p>
<h3>Training</h3>
<p>
  When ONNX Runtime Training is available, a trainable scalar <code>__train_C</code> is injected via a <code>Mul</code> on the first input of the tested node (initialized to 1.0).
  We generate artifacts (AdamW) and run a single optimization step with an MSE loss on the first output. Operators that complete this step are marked <strong>SUCCESS</strong>;
  explicitly skipped or unsupported patterns are <strong>SKIPPED</strong>; others are <strong>FAIL</strong>.
</p>

<h2>📦 Currently Supported Execution Providers (this opset)</h2>
<ul>
<li><a href="./CPU/" target="_blank">CPU</a></li>
<li><a href="./Intel%20-%20OpenVINO%E2%84%A2/" target="_blank">Intel - OpenVINO™</a></li>
<li><a href="./Intel%20-%20oneDNN/" target="_blank">Intel - oneDNN</a></li>
<li><a href="./NVIDIA%20-%20CUDA/" target="_blank">NVIDIA - CUDA</a></li>
<li><a href="./NVIDIA%20-%20TensorRT/" target="_blank">NVIDIA - TensorRT</a></li>
<li><a href="./Windows%20-%20DirectML/" target="_blank">Windows - DirectML</a></li>
</ul>

<h2>📊 Summary for Opset 20</h2>
<ul>
  <li><strong>CPU:</strong> Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz</li>
  <li><strong>GPU:</strong> NVIDIA GeForce RTX 2070</li>
  <li><strong>CUDA:</strong> 12.9 | <strong>cuDNN:</strong> 9.12.0 | <strong>TensorRT:</strong> 10.9.0</li>
  <li><strong>ONNX:</strong> 1.18.0 | <strong>ONNXRuntime:</strong> 1.23.0+cu125</li>
  <li><strong>OS:</strong> Windows 10</li>
</ul>

<h3>⚙️ Test Configuration</h3>
<ul>
  <li><strong>ONNX Opset version:</strong> 20</li>
  <li><strong>ONNX IR version:</strong> 10</li>
  <li><strong>Data types:</strong> Single dtype per node (usually <code>float32</code>).</li>
</ul>
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
      <th>Training</th>
    </tr>
  </thead>
  <tbody>
<tr><td><a href="./CPU/" target="_blank">CPU</a></td><td>149 (56%)</td><td>0 (0%)</td><td>149 (56%)</td><td>83 (31%)</td><td>0 (0%)</td><td>34 (13%)</td><td><strong>41 (27%)</strong></td></tr>
<tr><td><a href="./Intel%20-%20OpenVINO%E2%84%A2/" target="_blank">Intel - OpenVINO™</a></td><td>91 (59%)</td><td>56 (37%)</td><td>147 (96%)</td><td>6 (4%)</td><td>0 (0%)</td><td>0 (0%)</td><td><strong>0 (0%)</strong></td></tr>
<tr><td><a href="./Intel%20-%20oneDNN/" target="_blank">Intel - oneDNN</a></td><td>36 (24%)</td><td>113 (74%)</td><td>149 (97%)</td><td>4 (3%)</td><td>0 (0%)</td><td>0 (0%)</td><td><strong>0 (0%)</strong></td></tr>
<tr><td><a href="./NVIDIA%20-%20CUDA/" target="_blank">NVIDIA - CUDA</a></td><td>100 (38%)</td><td>49 (19%)</td><td>149 (56%)</td><td>81 (31%)</td><td>0 (0%)</td><td>34 (13%)</td><td><strong>43 (28%)</strong></td></tr>
<tr><td><a href="./NVIDIA%20-%20TensorRT/" target="_blank">NVIDIA - TensorRT</a></td><td>87 (57%)</td><td>60 (39%)</td><td>147 (96%)</td><td>6 (4%)</td><td>0 (0%)</td><td>0 (0%)</td><td><strong>0 (0%)</strong></td></tr>
<tr><td><a href="./Windows%20-%20DirectML/" target="_blank">Windows - DirectML</a></td><td>113 (74%)</td><td>35 (23%)</td><td>148 (97%)</td><td>5 (3%)</td><td>0 (0%)</td><td>0 (0%)</td><td><strong>0 (0%)</strong></td></tr>
</tbody></table>
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
      <th>Training</th>
    </tr>
  </thead>
  <tbody>
<tr><td><a href="./CPU/" target="_blank">CPU</a></td><td>59 (29%)</td><td>0 (0%)</td><td>59 (29%)</td><td>105 (51%)</td><td>7 (3%)</td><td>36 (17%)</td><td><strong>7 (7%)</strong></td></tr>
<tr><td><a href="./Intel%20-%20OpenVINO%E2%84%A2/" target="_blank">Intel - OpenVINO™</a></td><td>15 (14%)</td><td>41 (38%)</td><td>56 (52%)</td><td>44 (41%)</td><td>7 (7%)</td><td>0 (0%)</td><td><strong>0 (0%)</strong></td></tr>
<tr><td><a href="./Intel%20-%20oneDNN/" target="_blank">Intel - oneDNN</a></td><td>6 (6%)</td><td>52 (49%)</td><td>58 (54%)</td><td>42 (39%)</td><td>7 (7%)</td><td>0 (0%)</td><td><strong>0 (0%)</strong></td></tr>
<tr><td><a href="./NVIDIA%20-%20CUDA/" target="_blank">NVIDIA - CUDA</a></td><td>52 (25%)</td><td>34 (16%)</td><td>86 (42%)</td><td>78 (38%)</td><td>7 (3%)</td><td>36 (17%)</td><td><strong>7 (7%)</strong></td></tr>
<tr><td><a href="./NVIDIA%20-%20TensorRT/" target="_blank">NVIDIA - TensorRT</a></td><td>6 (6%)</td><td>78 (73%)</td><td>84 (79%)</td><td>16 (15%)</td><td>7 (7%)</td><td>0 (0%)</td><td><strong>0 (0%)</strong></td></tr>
<tr><td><a href="./Windows%20-%20DirectML/" target="_blank">Windows - DirectML</a></td><td>27 (25%)</td><td>33 (31%)</td><td>60 (56%)</td><td>40 (37%)</td><td>7 (7%)</td><td>0 (0%)</td><td><strong>0 (0%)</strong></td></tr>
</tbody></table>

<h2>🧭 Related Tools</h2>
<p>
  For a complementary view on backend compliance, see the official
  <a href="https://onnx.ai/backend-scoreboard/" target="_blank"><strong>ONNX Backend Scoreboard</strong></a>.
</p>

<h2>🤝 Maintainer</h2>
<p>
  Maintained by <strong><a href="https://graiphic.io/" target="_blank">Graiphic</a></strong> as part of the
  <a href="https://graiphic.io/download/" target="_blank"><strong>SOTA</strong></a> initiative.
  Contributions and feedback are welcome.
</p>

</div>