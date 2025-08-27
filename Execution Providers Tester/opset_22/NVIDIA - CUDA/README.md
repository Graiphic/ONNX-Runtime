# ONNXRuntime Test Results — Provider: `CUDAExecutionProvider`

**Test Date:** 2025-08-27 12:01:05

## Test Methodology

### Inference
Each ONNX operator is tested individually using a minimal ONNX model containing only that specific node. This ensures a focused and isolated evaluation of operator support for the selected Execution Provider.

### Training
To validate training (backward) support with ONNX Runtime Training, we **inject a `Mul` node** just before the tested operator: the **first input** of the tested node is multiplied by a **trainable scalar** `__train_C` (initialized to **1.0** so the forward values remain unchanged). We focus on the first input because it generally carries the data flow; for symmetric binary ops (e.g., `Add`), if training works on the first path it usually works on the others as well.

We then generate ONNX Runtime **training artifacts** (AdamW), run an inference once to **patch output shapes** if needed, feed a **target equal to the model’s own output** (MSE loss on the first output), and perform **one optimization step**. A node is marked **SUCCESS** when this step completes; **SKIPPED** for explicitly skipped ops (e.g., some recurrent ops like GRU/LSTM) or unsupported input types for this method; otherwise it is **FAIL**.

### Test Configuration

- **ONNX Opset version:** 22
- **ONNX IR version:** 10
- **Data types:** Only one type is tested per node. This is usually `float32`, unless the node does not support it — in which case a compatible type is selected.

> **Note:** Some ONNX nodes may not be available on the selected Execution Provider (EP) for opset version 22. This can lead to fallback behavior even though these nodes were supported in earlier opset versions. This occurs because ONNX Runtime teams may not have implemented or updated certain operators for the latest opset. As a result, test outcomes can vary depending on both the ONNX opset version and the ONNX Runtime version used.

## Environment and Installation Details

- **ONNX version:** 1.18.0
- **ONNXRuntime version:** 1.23.0+cu125
- **Target provider:** CUDAExecutionProvider
- **Installation command:**
```bash
pip install onnxruntime-gpu
```
### Hardware and Software Versions

- **CPU:** Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz
- **GPU(s):** NVIDIA GeForce RTX 2070
- **CUDA version:** 12.9
- **cuDNN version:** 9.12.0

## Basic ONNX Nodes

| ONNX Node | Status | Training |
|:---------:|:------:|:--------:|
| [`Add`](https://onnx.ai/onnx/operators/onnx__Add.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`AffineGrid`](https://onnx.ai/onnx/operators/onnx__AffineGrid.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`And`](https://onnx.ai/onnx/operators/onnx__And.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`ArgMax`](https://onnx.ai/onnx/operators/onnx__ArgMax.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`ArgMin`](https://onnx.ai/onnx/operators/onnx__ArgMin.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`AveragePool`](https://onnx.ai/onnx/operators/onnx__AveragePool.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`BatchNormalization`](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Bernoulli`](https://onnx.ai/onnx/operators/onnx__Bernoulli.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`BitShift`](https://onnx.ai/onnx/operators/onnx__BitShift.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`BitwiseAnd`](https://onnx.ai/onnx/operators/onnx__BitwiseAnd.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`BitwiseNot`](https://onnx.ai/onnx/operators/onnx__BitwiseNot.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`BitwiseOr`](https://onnx.ai/onnx/operators/onnx__BitwiseOr.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`BitwiseXor`](https://onnx.ai/onnx/operators/onnx__BitwiseXor.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`BlackmanWindow`](https://onnx.ai/onnx/operators/onnx__BlackmanWindow.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`Cast`](https://onnx.ai/onnx/operators/onnx__Cast.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`CastLike`](https://onnx.ai/onnx/operators/onnx__CastLike.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`CenterCropPad`](https://onnx.ai/onnx/operators/onnx__CenterCropPad.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Clip`](https://onnx.ai/onnx/operators/onnx__Clip.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Col2Im`](https://onnx.ai/onnx/operators/onnx__Col2Im.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Compress`](https://onnx.ai/onnx/operators/onnx__Compress.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Concat`](https://onnx.ai/onnx/operators/onnx__Concat.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ConcatFromSequence`](https://onnx.ai/onnx/operators/onnx__ConcatFromSequence.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`Constant`](https://onnx.ai/onnx/operators/onnx__Constant.html) | ![UNKNOWN](https://img.shields.io/badge/UNKNOWN-AAAAAA?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`ConstantOfShape`](https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`Conv`](https://onnx.ai/onnx/operators/onnx__Conv.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ConvInteger`](https://onnx.ai/onnx/operators/onnx__ConvInteger.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`ConvTranspose`](https://onnx.ai/onnx/operators/onnx__ConvTranspose.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`CumSum`](https://onnx.ai/onnx/operators/onnx__CumSum.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`DFT`](https://onnx.ai/onnx/operators/onnx__DFT.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`DeformConv`](https://onnx.ai/onnx/operators/onnx__DeformConv.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`DepthToSpace`](https://onnx.ai/onnx/operators/onnx__DepthToSpace.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`DequantizeLinear`](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`Det`](https://onnx.ai/onnx/operators/onnx__Det.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Div`](https://onnx.ai/onnx/operators/onnx__Div.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Dropout`](https://onnx.ai/onnx/operators/onnx__Dropout.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`DynamicQuantizeLinear`](https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Einsum`](https://onnx.ai/onnx/operators/onnx__Einsum.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Equal`](https://onnx.ai/onnx/operators/onnx__Equal.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Expand`](https://onnx.ai/onnx/operators/onnx__Expand.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`EyeLike`](https://onnx.ai/onnx/operators/onnx__EyeLike.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Flatten`](https://onnx.ai/onnx/operators/onnx__Flatten.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`GRU`](https://onnx.ai/onnx/operators/onnx__GRU.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`Gather`](https://onnx.ai/onnx/operators/onnx__Gather.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`GatherElements`](https://onnx.ai/onnx/operators/onnx__GatherElements.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`GatherND`](https://onnx.ai/onnx/operators/onnx__GatherND.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Gelu`](https://onnx.ai/onnx/operators/onnx__Gelu.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Gemm`](https://onnx.ai/onnx/operators/onnx__Gemm.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`GlobalAveragePool`](https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`GlobalMaxPool`](https://onnx.ai/onnx/operators/onnx__GlobalMaxPool.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Greater`](https://onnx.ai/onnx/operators/onnx__Greater.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`GreaterOrEqual`](https://onnx.ai/onnx/operators/onnx__GreaterOrEqual.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`GridSample`](https://onnx.ai/onnx/operators/onnx__GridSample.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`GroupNormalization`](https://onnx.ai/onnx/operators/onnx__GroupNormalization.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`HammingWindow`](https://onnx.ai/onnx/operators/onnx__HammingWindow.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`HannWindow`](https://onnx.ai/onnx/operators/onnx__HannWindow.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`HardSigmoid`](https://onnx.ai/onnx/operators/onnx__HardSigmoid.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`HardSwish`](https://onnx.ai/onnx/operators/onnx__HardSwish.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Hardmax`](https://onnx.ai/onnx/operators/onnx__Hardmax.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Identity`](https://onnx.ai/onnx/operators/onnx__Identity.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`If`](https://onnx.ai/onnx/operators/onnx__If.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`ImageDecoder`](https://onnx.ai/onnx/operators/onnx__ImageDecoder.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`InstanceNormalization`](https://onnx.ai/onnx/operators/onnx__InstanceNormalization.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`IsInf`](https://onnx.ai/onnx/operators/onnx__IsInf.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`IsNaN`](https://onnx.ai/onnx/operators/onnx__IsNaN.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`LRN`](https://onnx.ai/onnx/operators/onnx__LRN.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`LSTM`](https://onnx.ai/onnx/operators/onnx__LSTM.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`LayerNormalization`](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Less`](https://onnx.ai/onnx/operators/onnx__Less.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`LessOrEqual`](https://onnx.ai/onnx/operators/onnx__LessOrEqual.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Loop`](https://onnx.ai/onnx/operators/onnx__Loop.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`LpNormalization`](https://onnx.ai/onnx/operators/onnx__LpNormalization.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`LpPool`](https://onnx.ai/onnx/operators/onnx__LpPool.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`MatMul`](https://onnx.ai/onnx/operators/onnx__MatMul.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`MatMulInteger`](https://onnx.ai/onnx/operators/onnx__MatMulInteger.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`Max`](https://onnx.ai/onnx/operators/onnx__Max.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`MaxPool`](https://onnx.ai/onnx/operators/onnx__MaxPool.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`MaxRoiPool`](https://onnx.ai/onnx/operators/onnx__MaxRoiPool.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`MaxUnpool`](https://onnx.ai/onnx/operators/onnx__MaxUnpool.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Mean`](https://onnx.ai/onnx/operators/onnx__Mean.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`MeanVarianceNormalization`](https://onnx.ai/onnx/operators/onnx__MeanVarianceNormalization.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`MelWeightMatrix`](https://onnx.ai/onnx/operators/onnx__MelWeightMatrix.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`Min`](https://onnx.ai/onnx/operators/onnx__Min.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Mod`](https://onnx.ai/onnx/operators/onnx__Mod.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Mul`](https://onnx.ai/onnx/operators/onnx__Mul.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Multinomial`](https://onnx.ai/onnx/operators/onnx__Multinomial.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`NegativeLogLikelihoodLoss`](https://onnx.ai/onnx/operators/onnx__NegativeLogLikelihoodLoss.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`NonMaxSuppression`](https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`NonZero`](https://onnx.ai/onnx/operators/onnx__NonZero.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Not`](https://onnx.ai/onnx/operators/onnx__Not.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`Not`](https://onnx.ai/onnx/operators/onnx__Not.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`OneHot`](https://onnx.ai/onnx/operators/onnx__OneHot.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`Optional`](https://onnx.ai/onnx/operators/onnx__Optional.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`OptionalGetElement`](https://onnx.ai/onnx/operators/onnx__OptionalGetElement.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`OptionalHasElement`](https://onnx.ai/onnx/operators/onnx__OptionalHasElement.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Or`](https://onnx.ai/onnx/operators/onnx__Or.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`PRelu`](https://onnx.ai/onnx/operators/onnx__PRelu.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Pad`](https://onnx.ai/onnx/operators/onnx__Pad.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Pow`](https://onnx.ai/onnx/operators/onnx__Pow.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`QLinearConv`](https://onnx.ai/onnx/operators/onnx__QLinearConv.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`QLinearMatMul`](https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`QuantizeLinear`](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`RNN`](https://onnx.ai/onnx/operators/onnx__RNN.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`RandomNormal`](https://onnx.ai/onnx/operators/onnx__RandomNormal.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`RandomNormalLike`](https://onnx.ai/onnx/operators/onnx__RandomNormalLike.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`RandomUniform`](https://onnx.ai/onnx/operators/onnx__RandomUniform.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`RandomUniformLike`](https://onnx.ai/onnx/operators/onnx__RandomUniformLike.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Range`](https://onnx.ai/onnx/operators/onnx__Range.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`ReduceL1`](https://onnx.ai/onnx/operators/onnx__ReduceL1.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`ReduceL2`](https://onnx.ai/onnx/operators/onnx__ReduceL2.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceLogSum`](https://onnx.ai/onnx/operators/onnx__ReduceLogSum.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`ReduceLogSumExp`](https://onnx.ai/onnx/operators/onnx__ReduceLogSumExp.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceMax`](https://onnx.ai/onnx/operators/onnx__ReduceMax.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceMean`](https://onnx.ai/onnx/operators/onnx__ReduceMean.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceMin`](https://onnx.ai/onnx/operators/onnx__ReduceMin.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`ReduceProd`](https://onnx.ai/onnx/operators/onnx__ReduceProd.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`ReduceSum`](https://onnx.ai/onnx/operators/onnx__ReduceSum.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceSumSquare`](https://onnx.ai/onnx/operators/onnx__ReduceSumSquare.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`RegexFullMatch`](https://onnx.ai/onnx/operators/onnx__RegexFullMatch.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`Reshape`](https://onnx.ai/onnx/operators/onnx__Reshape.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Resize`](https://onnx.ai/onnx/operators/onnx__Resize.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReverseSequence`](https://onnx.ai/onnx/operators/onnx__ReverseSequence.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`RoiAlign`](https://onnx.ai/onnx/operators/onnx__RoiAlign.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`STFT`](https://onnx.ai/onnx/operators/onnx__STFT.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Scan`](https://onnx.ai/onnx/operators/onnx__Scan.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`ScatterElements`](https://onnx.ai/onnx/operators/onnx__ScatterElements.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ScatterND`](https://onnx.ai/onnx/operators/onnx__ScatterND.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`SequenceAt`](https://onnx.ai/onnx/operators/onnx__SequenceAt.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`SequenceConstruct`](https://onnx.ai/onnx/operators/onnx__SequenceConstruct.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`SequenceEmpty`](https://onnx.ai/onnx/operators/onnx__SequenceEmpty.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`SequenceErase`](https://onnx.ai/onnx/operators/onnx__SequenceErase.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`SequenceInsert`](https://onnx.ai/onnx/operators/onnx__SequenceInsert.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`SequenceLength`](https://onnx.ai/onnx/operators/onnx__SequenceLength.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`SequenceMap`](https://onnx.ai/onnx/operators/onnx__SequenceMap.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`Shape`](https://onnx.ai/onnx/operators/onnx__Shape.html) | ![UNKNOWN](https://img.shields.io/badge/UNKNOWN-AAAAAA?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Size`](https://onnx.ai/onnx/operators/onnx__Size.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Slice`](https://onnx.ai/onnx/operators/onnx__Slice.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`SoftmaxCrossEntropyLoss`](https://onnx.ai/onnx/operators/onnx__SoftmaxCrossEntropyLoss.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`SpaceToDepth`](https://onnx.ai/onnx/operators/onnx__SpaceToDepth.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Split`](https://onnx.ai/onnx/operators/onnx__Split.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`SplitToSequence`](https://onnx.ai/onnx/operators/onnx__SplitToSequence.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`Squeeze`](https://onnx.ai/onnx/operators/onnx__Squeeze.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`StringConcat`](https://onnx.ai/onnx/operators/onnx__StringConcat.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`StringNormalizer`](https://onnx.ai/onnx/operators/onnx__StringNormalizer.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`StringSplit`](https://onnx.ai/onnx/operators/onnx__StringSplit.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`Sub`](https://onnx.ai/onnx/operators/onnx__Sub.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Sum`](https://onnx.ai/onnx/operators/onnx__Sum.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`TfIdfVectorizer`](https://onnx.ai/onnx/operators/onnx__TfIdfVectorizer.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`Tile`](https://onnx.ai/onnx/operators/onnx__Tile.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`TopK`](https://onnx.ai/onnx/operators/onnx__TopK.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Transpose`](https://onnx.ai/onnx/operators/onnx__Transpose.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Trilu`](https://onnx.ai/onnx/operators/onnx__Trilu.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Unique`](https://onnx.ai/onnx/operators/onnx__Unique.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`Unsqueeze`](https://onnx.ai/onnx/operators/onnx__Unsqueeze.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Where`](https://onnx.ai/onnx/operators/onnx__Where.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`Xor`](https://onnx.ai/onnx/operators/onnx__Xor.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |

### Statistics
- **Total nodes tested:** 155
- **Executable directly (SUCCESS):** 71 (45.8%)
- **Executable directly (SUCCESS with complexification):** 0 (0.0%)
- **Executable via FALLBACK:** 71 (45.8%)
- **UNKNOWN (no Node event):** 2 (1.3%)
- **NOT TESTED:** 0 (0.0%)
- **SKIPPED:** 0 (0.0%)
- **FAIL:** 11 (7.1%)

![Pie Chart](./stats_CUDAExecutionProvider_basic.png)

### Training Summary (Basic ONNX Nodes)
- **Total nodes tested (training):** 154
- **SUCCESS:** 42
- **FAIL:** 78
- **SKIPPED:** 34

![Training Pie Chart — Basic](./training_stats_CUDAExecutionProvider_basic.png)

## Microsoft Custom Nodes

| ONNX Node | Status | Training |
|:---------:|:------:|:--------:|
| [`com.microsoft.Attention`](https://onnx.ai/onnx/operators/onnx__com.microsoft.Attention.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.BeamSearch`](https://onnx.ai/onnx/operators/onnx__com.microsoft.BeamSearch.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.BiasAdd`](https://onnx.ai/onnx/operators/onnx__com.microsoft.BiasAdd.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.BiasDropout`](https://onnx.ai/onnx/operators/onnx__com.microsoft.BiasDropout.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.BiasGelu`](https://onnx.ai/onnx/operators/onnx__com.microsoft.BiasGelu.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`com.microsoft.BiasSoftmax`](https://onnx.ai/onnx/operators/onnx__com.microsoft.BiasSoftmax.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.BiasSplitGelu`](https://onnx.ai/onnx/operators/onnx__com.microsoft.BiasSplitGelu.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.BifurcationDetector`](https://onnx.ai/onnx/operators/onnx__com.microsoft.BifurcationDetector.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.BitmaskBiasDropout`](https://onnx.ai/onnx/operators/onnx__com.microsoft.BitmaskBiasDropout.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.BitmaskDropout`](https://onnx.ai/onnx/operators/onnx__com.microsoft.BitmaskDropout.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.CDist`](https://onnx.ai/onnx/operators/onnx__com.microsoft.CDist.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.ComplexMul`](https://onnx.ai/onnx/operators/onnx__com.microsoft.ComplexMul.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.ComplexMulConj`](https://onnx.ai/onnx/operators/onnx__com.microsoft.ComplexMulConj.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.ConvTransposeWithDynamicPads`](https://onnx.ai/onnx/operators/onnx__com.microsoft.ConvTransposeWithDynamicPads.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.CropAndResize`](https://onnx.ai/onnx/operators/onnx__com.microsoft.CropAndResize.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.DecoderAttention`](https://onnx.ai/onnx/operators/onnx__com.microsoft.DecoderAttention.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.DecoderMaskedMultiHeadAttention`](https://onnx.ai/onnx/operators/onnx__com.microsoft.DecoderMaskedMultiHeadAttention.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.DecoderMaskedSelfAttention`](https://onnx.ai/onnx/operators/onnx__com.microsoft.DecoderMaskedSelfAttention.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.DequantizeBFP`](https://onnx.ai/onnx/operators/onnx__com.microsoft.DequantizeBFP.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.DequantizeLinear`](https://onnx.ai/onnx/operators/onnx__com.microsoft.DequantizeLinear.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.DequantizeWithOrder`](https://onnx.ai/onnx/operators/onnx__com.microsoft.DequantizeWithOrder.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.DynamicQuantizeLSTM`](https://onnx.ai/onnx/operators/onnx__com.microsoft.DynamicQuantizeLSTM.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.DynamicQuantizeMatMul`](https://onnx.ai/onnx/operators/onnx__com.microsoft.DynamicQuantizeMatMul.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.DynamicTimeWarping`](https://onnx.ai/onnx/operators/onnx__com.microsoft.DynamicTimeWarping.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.EmbedLayerNormalization`](https://onnx.ai/onnx/operators/onnx__com.microsoft.EmbedLayerNormalization.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.ExpandDims`](https://onnx.ai/onnx/operators/onnx__com.microsoft.ExpandDims.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.FastGelu`](https://onnx.ai/onnx/operators/onnx__com.microsoft.FastGelu.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`com.microsoft.FusedConv`](https://onnx.ai/onnx/operators/onnx__com.microsoft.FusedConv.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.FusedGemm`](https://onnx.ai/onnx/operators/onnx__com.microsoft.FusedGemm.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.FusedMatMul`](https://onnx.ai/onnx/operators/onnx__com.microsoft.FusedMatMul.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.FusedMatMulActivation`](https://onnx.ai/onnx/operators/onnx__com.microsoft.FusedMatMulActivation.html) | ![NOT TESTED](https://img.shields.io/badge/NOT%20TESTED-7777CC?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.GatedRelativePositionBias`](https://onnx.ai/onnx/operators/onnx__com.microsoft.GatedRelativePositionBias.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.GatherBlockQuantized`](https://onnx.ai/onnx/operators/onnx__com.microsoft.GatherBlockQuantized.html) | ![NOT TESTED](https://img.shields.io/badge/NOT%20TESTED-7777CC?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.GatherND`](https://onnx.ai/onnx/operators/onnx__com.microsoft.GatherND.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.Gelu`](https://onnx.ai/onnx/operators/onnx__com.microsoft.Gelu.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`com.microsoft.GemmFastGelu`](https://onnx.ai/onnx/operators/onnx__com.microsoft.GemmFastGelu.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.GemmFloat8`](https://onnx.ai/onnx/operators/onnx__com.microsoft.GemmFloat8.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.GemmaRotaryEmbedding`](https://onnx.ai/onnx/operators/onnx__com.microsoft.GemmaRotaryEmbedding.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.GreedySearch`](https://onnx.ai/onnx/operators/onnx__com.microsoft.GreedySearch.html) | ![NOT TESTED](https://img.shields.io/badge/NOT%20TESTED-7777CC?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.GridSample`](https://onnx.ai/onnx/operators/onnx__com.microsoft.GridSample.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.GroupNorm`](https://onnx.ai/onnx/operators/onnx__com.microsoft.GroupNorm.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.GroupQueryAttention`](https://onnx.ai/onnx/operators/onnx__com.microsoft.GroupQueryAttention.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.Inverse`](https://onnx.ai/onnx/operators/onnx__com.microsoft.Inverse.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.Irfft`](https://onnx.ai/onnx/operators/onnx__com.microsoft.Irfft.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.LongformerAttention`](https://onnx.ai/onnx/operators/onnx__com.microsoft.LongformerAttention.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.MatMulBnb4`](https://onnx.ai/onnx/operators/onnx__com.microsoft.MatMulBnb4.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`com.microsoft.MatMulFpQ4`](https://onnx.ai/onnx/operators/onnx__com.microsoft.MatMulFpQ4.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.MatMulInteger16`](https://onnx.ai/onnx/operators/onnx__com.microsoft.MatMulInteger16.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.MatMulIntegerToFloat`](https://onnx.ai/onnx/operators/onnx__com.microsoft.MatMulIntegerToFloat.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.MatMulNBits`](https://onnx.ai/onnx/operators/onnx__com.microsoft.MatMulNBits.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.MaxpoolWithMask`](https://onnx.ai/onnx/operators/onnx__com.microsoft.MaxpoolWithMask.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.MoE`](https://onnx.ai/onnx/operators/onnx__com.microsoft.MoE.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.MulInteger`](https://onnx.ai/onnx/operators/onnx__com.microsoft.MulInteger.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.MultiHeadAttention`](https://onnx.ai/onnx/operators/onnx__com.microsoft.MultiHeadAttention.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.MurmurHash3`](https://onnx.ai/onnx/operators/onnx__com.microsoft.MurmurHash3.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.NGramRepeatBlock`](https://onnx.ai/onnx/operators/onnx__com.microsoft.NGramRepeatBlock.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.NhwcConv`](https://onnx.ai/onnx/operators/onnx__com.microsoft.NhwcConv.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.NhwcFusedConv`](https://onnx.ai/onnx/operators/onnx__com.microsoft.NhwcFusedConv.html) | ![NOT TESTED](https://img.shields.io/badge/NOT%20TESTED-7777CC?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.NhwcMaxPool`](https://onnx.ai/onnx/operators/onnx__com.microsoft.NhwcMaxPool.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.PackedAttention`](https://onnx.ai/onnx/operators/onnx__com.microsoft.PackedAttention.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.PackedMultiHeadAttention`](https://onnx.ai/onnx/operators/onnx__com.microsoft.PackedMultiHeadAttention.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.Pad`](https://onnx.ai/onnx/operators/onnx__com.microsoft.Pad.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`com.microsoft.QAttention`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QAttention.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.QGemm`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QGemm.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.QLinearAdd`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QLinearAdd.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.QLinearAveragePool`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QLinearAveragePool.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.QLinearConcat`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QLinearConcat.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.QLinearConv`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QLinearConv.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.QLinearGlobalAveragePool`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QLinearGlobalAveragePool.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.QLinearLeakyRelu`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QLinearLeakyRelu.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.QLinearMul`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QLinearMul.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.QLinearReduceMean`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QLinearReduceMean.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.QLinearSigmoid`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QLinearSigmoid.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.QLinearSoftmax`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QLinearSoftmax.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.QLinearWhere`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QLinearWhere.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.QMoE`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QMoE.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.QOrderedAttention`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QOrderedAttention.html) | ![NOT TESTED](https://img.shields.io/badge/NOT%20TESTED-7777CC?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.QOrderedGelu`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QOrderedGelu.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.QOrderedLayerNormalization`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QOrderedLayerNormalization.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.QOrderedLongformerAttention`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QOrderedLongformerAttention.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.QOrderedMatMul`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QOrderedMatMul.html) | ![NOT TESTED](https://img.shields.io/badge/NOT%20TESTED-7777CC?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.QuantizeBFP`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QuantizeBFP.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.QuantizeLinear`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QuantizeLinear.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.QuantizeWithOrder`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QuantizeWithOrder.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.QuickGelu`](https://onnx.ai/onnx/operators/onnx__com.microsoft.QuickGelu.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`com.microsoft.Range`](https://onnx.ai/onnx/operators/onnx__com.microsoft.Range.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.ReduceSumInteger`](https://onnx.ai/onnx/operators/onnx__com.microsoft.ReduceSumInteger.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.RelativePositionBias`](https://onnx.ai/onnx/operators/onnx__com.microsoft.RelativePositionBias.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.RemovePadding`](https://onnx.ai/onnx/operators/onnx__com.microsoft.RemovePadding.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.RestorePadding`](https://onnx.ai/onnx/operators/onnx__com.microsoft.RestorePadding.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.Rfft`](https://onnx.ai/onnx/operators/onnx__com.microsoft.Rfft.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.RotaryEmbedding`](https://onnx.ai/onnx/operators/onnx__com.microsoft.RotaryEmbedding.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.SampleOp`](https://onnx.ai/onnx/operators/onnx__com.microsoft.SampleOp.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.Sampling`](https://onnx.ai/onnx/operators/onnx__com.microsoft.Sampling.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.SkipGroupNorm`](https://onnx.ai/onnx/operators/onnx__com.microsoft.SkipGroupNorm.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.SkipLayerNormalization`](https://onnx.ai/onnx/operators/onnx__com.microsoft.SkipLayerNormalization.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.SkipSimplifiedLayerNormalization`](https://onnx.ai/onnx/operators/onnx__com.microsoft.SkipSimplifiedLayerNormalization.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.Snpe`](https://onnx.ai/onnx/operators/onnx__com.microsoft.Snpe.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.SparseAttention`](https://onnx.ai/onnx/operators/onnx__com.microsoft.SparseAttention.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.SparseToDenseMatMul`](https://onnx.ai/onnx/operators/onnx__com.microsoft.SparseToDenseMatMul.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.Tokenizer`](https://onnx.ai/onnx/operators/onnx__com.microsoft.Tokenizer.html) | ![FALLBACK](https://img.shields.io/badge/FALLBACK-FFAA00?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.TorchEmbedding`](https://onnx.ai/onnx/operators/onnx__com.microsoft.TorchEmbedding.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.TransposeMatMul`](https://onnx.ai/onnx/operators/onnx__com.microsoft.TransposeMatMul.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.Trilu`](https://onnx.ai/onnx/operators/onnx__com.microsoft.Trilu.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`com.microsoft.UnfoldTensor`](https://onnx.ai/onnx/operators/onnx__com.microsoft.UnfoldTensor.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |
| [`com.microsoft.Unique`](https://onnx.ai/onnx/operators/onnx__com.microsoft.Unique.html) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) | ![SKIPPED](https://img.shields.io/badge/SKIPPED-999999?style=flat&logoColor=white) |
| [`com.microsoft.WhisperBeamSearch`](https://onnx.ai/onnx/operators/onnx__com.microsoft.WhisperBeamSearch.html) | ![NOT TESTED](https://img.shields.io/badge/NOT%20TESTED-7777CC?style=flat&logoColor=white) | ![FAIL](https://img.shields.io/badge/FAIL-FF0000?style=flat&logoColor=white) |

### Statistics
- **Total nodes tested:** 107
- **Executable directly (SUCCESS):** 52 (48.6%)
- **Executable directly (SUCCESS with complexification):** 0 (0.0%)
- **Executable via FALLBACK:** 34 (31.8%)
- **UNKNOWN (no Node event):** 0 (0.0%)
- **NOT TESTED:** 7 (6.5%)
- **SKIPPED:** 0 (0.0%)
- **FAIL:** 14 (13.1%)

![Pie Chart](./stats_CUDAExecutionProvider_ms.png)

### Training Summary (Microsoft Custom Nodes)
- **Total nodes tested (training):** 107
- **SUCCESS:** 7
- **FAIL:** 64
- **SKIPPED:** 36

![Training Pie Chart — Microsoft](./training_stats_CUDAExecutionProvider_ms.png)

## Nodes not tested

These nodes couldn't be tested due to lack of valid minimal ONNX model.

`com.microsoft.FusedMatMulActivation`, `com.microsoft.GatherBlockQuantized`, `com.microsoft.GreedySearch`, `com.microsoft.NhwcFusedConv`, `com.microsoft.QOrderedAttention`, `com.microsoft.QOrderedMatMul`, `com.microsoft.WhisperBeamSearch`

## README Generation

This file was generated automatically by `report.py`.

- Generated ONNX models: `models/<provider>/`
- Profiling JSON files: `profiling/<provider>/`
- Scripts: `main.py`, `report.py`, `utils.py`, `ops/*`
_End of README_