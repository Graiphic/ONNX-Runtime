# Summary of ONNX Execution Provider Results

This document gathers all per-provider test results in one place.  Below is a table summarizing, for each Execution Provider (EP), how many nodes ran successfully on the provider directly (SUCCESS), how many fell back (FALLBACK), and how many failed (FAIL), each with its percentage of the total.  Refer to each EP’s own README for detailed charts and further statistics.

| Execution Provider | SUCCESS | FALLBACK | FAIL |
|:------------------:|:-------:|:--------:|:----:|
| CPU | 150 (100%) | 0 (0%) | 0 (0%) |
| Intel - OneDNN | 39 (26%) | 111 (74%) | 0 (0%) |
| Intel - OpenVINO™ | 65 (43%) | 83 (55%) | 2 (1%) |
| Nvidia - CUDA | 74 (49%) | 76 (51%) | 0 (0%) |
| Nvidia - TensorRT | 87 (58%) | 59 (39%) | 4 (3%) |
| Windows - DirectML | 100 (67%) | 49 (33%) | 1 (1%) |

## Testing Methodology

Each ONNX operator (node) is tested by dynamically building a minimal ONNX model containing only that single node.  The test harness then creates an ONNXRuntime inference session with profiling enabled, runs the model once, and inspects the generated profiling trace to determine whether the node was executed successfully.  From the profiling events, we also identify which Execution Provider (EP) actually processed the node—either directly on the target EP or via a fallback path.  If the node did not produce any 'Node' event in the profiler output, it is marked as UNKNOWN; otherwise, it is classified as SUCCESS (direct) or FALLBACK (if ONNXRuntime fell back to another EP).

## Hardware and Software

- **CPU:** Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz
- **GPU(s):** NVIDIA GeForce RTX 2070
- **CUDA version:** 12.5
- **cuDNN version:** 9.2.1
- **TensorRT version:** 10.9.0
- **ONNX version:** 1.18.0
- **ONNXRuntime version:** 1.23.0
- **Operating System (OS):** Windows 10

