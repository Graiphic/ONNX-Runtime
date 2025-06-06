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

