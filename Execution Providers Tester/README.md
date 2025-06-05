# Summary of ONNX Execution Provider Results

This document gathers all test results by Execution Provider (EP).  
Each EP has generated its own README with detailed statistics.  
Below, you will first find the hardware and software information used,  
followed by a summary table of the number of nodes that succeeded directly (SUCCESS), fell back (FALLBACK), or failed (FAIL) for each EP.

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

| Execution Provider | SUCCESS | FALLBACK | FAIL |
|:------------------:|:-------:|:--------:|:----:|
| CPU | 150 (100%) | 0 (0%) | 0 (0%) |
| Intel - OneDNN | 39 (26%) | 111 (74%) | 0 (0%) |
| Intel - OpenVINO™ | 65 (43%) | 83 (55%) | 2 (1%) |
| Nvidia - CUDA | 74 (49%) | 76 (51%) | 0 (0%) |
| Nvidia - TensorRT | 87 (58%) | 59 (39%) | 4 (3%) |
| Windows - DirectML | 100 (67%) | 49 (33%) | 1 (1%) |

