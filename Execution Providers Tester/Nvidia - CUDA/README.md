# ONNXRuntime Test Results — Provider: `CUDAExecutionProvider`

**Test Date:** 2025-06-05 09:16:34

## Environment and Installation Details

- **ONNX version:** 1.18.0
- **ONNXRuntime version:** 1.23.0
- **Target provider:** CUDAExecutionProvider
- **Installation command for this provider:**
```bash
pip install onnxruntime-gpu
```

### Hardware and Software Versions

- **CPU:** Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz
- **GPU(s):** NVIDIA GeForce RTX 2070
- **CUDA version:** 12.5
- **cuDNN version:** 9.2.1

## Node Details

| ONNX Node | Status |
|:---------:|:------:|
| [`Add`](https://onnx.ai/onnx/operators/onnx__Add.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Div`](https://onnx.ai/onnx/operators/onnx__Div.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Sub`](https://onnx.ai/onnx/operators/onnx__Sub.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Mul`](https://onnx.ai/onnx/operators/onnx__Mul.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Pow`](https://onnx.ai/onnx/operators/onnx__Pow.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Mod`](https://onnx.ai/onnx/operators/onnx__Mod.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`BitwiseAnd`](https://onnx.ai/onnx/operators/onnx__BitwiseAnd.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`BitwiseOr`](https://onnx.ai/onnx/operators/onnx__BitwiseOr.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`BitwiseXor`](https://onnx.ai/onnx/operators/onnx__BitwiseXor.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`And`](https://onnx.ai/onnx/operators/onnx__And.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Or`](https://onnx.ai/onnx/operators/onnx__Or.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Xor`](https://onnx.ai/onnx/operators/onnx__Xor.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceSum`](https://onnx.ai/onnx/operators/onnx__ReduceSum.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceMean`](https://onnx.ai/onnx/operators/onnx__ReduceMean.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceMax`](https://onnx.ai/onnx/operators/onnx__ReduceMax.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`ReduceMin`](https://onnx.ai/onnx/operators/onnx__ReduceMin.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`ReduceProd`](https://onnx.ai/onnx/operators/onnx__ReduceProd.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceL1`](https://onnx.ai/onnx/operators/onnx__ReduceL1.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceL2`](https://onnx.ai/onnx/operators/onnx__ReduceL2.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceLogSum`](https://onnx.ai/onnx/operators/onnx__ReduceLogSum.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceLogSumExp`](https://onnx.ai/onnx/operators/onnx__ReduceLogSumExp.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ReduceSumSquare`](https://onnx.ai/onnx/operators/onnx__ReduceSumSquare.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Equal`](https://onnx.ai/onnx/operators/onnx__Equal.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Not`](https://onnx.ai/onnx/operators/onnx__Not.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Greater`](https://onnx.ai/onnx/operators/onnx__Greater.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Less`](https://onnx.ai/onnx/operators/onnx__Less.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`GreaterOrEqual`](https://onnx.ai/onnx/operators/onnx__GreaterOrEqual.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`LessOrEqual`](https://onnx.ai/onnx/operators/onnx__LessOrEqual.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Not`](https://onnx.ai/onnx/operators/onnx__Not.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`IsNaN`](https://onnx.ai/onnx/operators/onnx__IsNaN.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`IsInf`](https://onnx.ai/onnx/operators/onnx__IsInf.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Max`](https://onnx.ai/onnx/operators/onnx__Max.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Min`](https://onnx.ai/onnx/operators/onnx__Min.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Mean`](https://onnx.ai/onnx/operators/onnx__Mean.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Sum`](https://onnx.ai/onnx/operators/onnx__Sum.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ArgMin`](https://onnx.ai/onnx/operators/onnx__ArgMin.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ArgMax`](https://onnx.ai/onnx/operators/onnx__ArgMax.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Reshape`](https://onnx.ai/onnx/operators/onnx__Reshape.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Expand`](https://onnx.ai/onnx/operators/onnx__Expand.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Flatten`](https://onnx.ai/onnx/operators/onnx__Flatten.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Squeeze`](https://onnx.ai/onnx/operators/onnx__Squeeze.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Unsqueeze`](https://onnx.ai/onnx/operators/onnx__Unsqueeze.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Transpose`](https://onnx.ai/onnx/operators/onnx__Transpose.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Concat`](https://onnx.ai/onnx/operators/onnx__Concat.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Split`](https://onnx.ai/onnx/operators/onnx__Split.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Slice`](https://onnx.ai/onnx/operators/onnx__Slice.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Pad`](https://onnx.ai/onnx/operators/onnx__Pad.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Tile`](https://onnx.ai/onnx/operators/onnx__Tile.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Gather`](https://onnx.ai/onnx/operators/onnx__Gather.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`GatherND`](https://onnx.ai/onnx/operators/onnx__GatherND.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`GatherElements`](https://onnx.ai/onnx/operators/onnx__GatherElements.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ScatterElements`](https://onnx.ai/onnx/operators/onnx__ScatterElements.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ScatterND`](https://onnx.ai/onnx/operators/onnx__ScatterND.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Gemm`](https://onnx.ai/onnx/operators/onnx__Gemm.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`MatMul`](https://onnx.ai/onnx/operators/onnx__MatMul.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`AveragePool`](https://onnx.ai/onnx/operators/onnx__AveragePool.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`MaxPool`](https://onnx.ai/onnx/operators/onnx__MaxPool.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`GlobalMaxPool`](https://onnx.ai/onnx/operators/onnx__GlobalMaxPool.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`GlobalAveragePool`](https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Dropout`](https://onnx.ai/onnx/operators/onnx__Dropout.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`BatchNormalization`](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`InstanceNormalization`](https://onnx.ai/onnx/operators/onnx__InstanceNormalization.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`LayerNormalization`](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`GroupNormalization`](https://onnx.ai/onnx/operators/onnx__GroupNormalization.html) | ![SUCCESS WITH FALLBACK (via decomposition)](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK%20(via%20decomposition)-FF7700?style=flat&logoColor=white) |
| [`AffineGrid`](https://onnx.ai/onnx/operators/onnx__AffineGrid.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`GridSample`](https://onnx.ai/onnx/operators/onnx__GridSample.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Resize`](https://onnx.ai/onnx/operators/onnx__Resize.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Bernoulli`](https://onnx.ai/onnx/operators/onnx__Bernoulli.html) | ![SUCCESS WITH FALLBACK (via decomposition)](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK%20(via%20decomposition)-FF7700?style=flat&logoColor=white) |
| [`BitShift`](https://onnx.ai/onnx/operators/onnx__BitShift.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`BitwiseNot`](https://onnx.ai/onnx/operators/onnx__BitwiseNot.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`BlackmanWindow`](https://onnx.ai/onnx/operators/onnx__BlackmanWindow.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Cast`](https://onnx.ai/onnx/operators/onnx__Cast.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`CastLike`](https://onnx.ai/onnx/operators/onnx__CastLike.html) | ![SUCCESS WITH FALLBACK (via decomposition)](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK%20(via%20decomposition)-FF7700?style=flat&logoColor=white) |
| [`CenterCropPad`](https://onnx.ai/onnx/operators/onnx__CenterCropPad.html) | ![SUCCESS WITH FALLBACK (via decomposition)](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK%20(via%20decomposition)-FF7700?style=flat&logoColor=white) |
| [`Clip`](https://onnx.ai/onnx/operators/onnx__Clip.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Col2Im`](https://onnx.ai/onnx/operators/onnx__Col2Im.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Compress`](https://onnx.ai/onnx/operators/onnx__Compress.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`ConcatFromSequence`](https://onnx.ai/onnx/operators/onnx__ConcatFromSequence.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Constant`](https://onnx.ai/onnx/operators/onnx__Constant.html) | ![UNKNOWN (no Node event)](https://img.shields.io/badge/UNKNOWN%20(no%20Node%20event)-DEDEDE?style=flat&logoColor=white) |
| [`ConstantOfShape`](https://onnx.ai/onnx/operators/onnx__ConstantOfShape.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Conv`](https://onnx.ai/onnx/operators/onnx__Conv.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`ConvTranspose`](https://onnx.ai/onnx/operators/onnx__ConvTranspose.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`ConvInteger`](https://onnx.ai/onnx/operators/onnx__ConvInteger.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`CumSum`](https://onnx.ai/onnx/operators/onnx__CumSum.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`DFT`](https://onnx.ai/onnx/operators/onnx__DFT.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`DepthToSpace`](https://onnx.ai/onnx/operators/onnx__DepthToSpace.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`DequantizeLinear`](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Det`](https://onnx.ai/onnx/operators/onnx__Det.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`DynamicQuantizeLinear`](https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Einsum`](https://onnx.ai/onnx/operators/onnx__Einsum.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`EyeLike`](https://onnx.ai/onnx/operators/onnx__EyeLike.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`GRU`](https://onnx.ai/onnx/operators/onnx__GRU.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Gelu`](https://onnx.ai/onnx/operators/onnx__Gelu.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`HammingWindow`](https://onnx.ai/onnx/operators/onnx__HammingWindow.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`HannWindow`](https://onnx.ai/onnx/operators/onnx__HannWindow.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`HardSigmoid`](https://onnx.ai/onnx/operators/onnx__HardSigmoid.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`HardSwish`](https://onnx.ai/onnx/operators/onnx__HardSwish.html) | ![SUCCESS WITH FALLBACK (via decomposition)](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK%20(via%20decomposition)-FF7700?style=flat&logoColor=white) |
| [`Hardmax`](https://onnx.ai/onnx/operators/onnx__Hardmax.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Identity`](https://onnx.ai/onnx/operators/onnx__Identity.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`If`](https://onnx.ai/onnx/operators/onnx__If.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`LRN`](https://onnx.ai/onnx/operators/onnx__LRN.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`LSTM`](https://onnx.ai/onnx/operators/onnx__LSTM.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Loop`](https://onnx.ai/onnx/operators/onnx__Loop.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`LpNormalization`](https://onnx.ai/onnx/operators/onnx__LpNormalization.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`LpPool`](https://onnx.ai/onnx/operators/onnx__LpPool.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`MatMulInteger`](https://onnx.ai/onnx/operators/onnx__MatMulInteger.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`MaxRoiPool`](https://onnx.ai/onnx/operators/onnx__MaxRoiPool.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`MaxUnpool`](https://onnx.ai/onnx/operators/onnx__MaxUnpool.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`MeanVarianceNormalization`](https://onnx.ai/onnx/operators/onnx__MeanVarianceNormalization.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`MelWeightMatrix`](https://onnx.ai/onnx/operators/onnx__MelWeightMatrix.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Multinomial`](https://onnx.ai/onnx/operators/onnx__Multinomial.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`NegativeLogLikelihoodLoss`](https://onnx.ai/onnx/operators/onnx__NegativeLogLikelihoodLoss.html) | ![SUCCESS WITH FALLBACK (via decomposition)](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK%20(via%20decomposition)-FF7700?style=flat&logoColor=white) |
| [`NonMaxSuppression`](https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`NonZero`](https://onnx.ai/onnx/operators/onnx__NonZero.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`OneHot`](https://onnx.ai/onnx/operators/onnx__OneHot.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Optional`](https://onnx.ai/onnx/operators/onnx__Optional.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`OptionalGetElement`](https://onnx.ai/onnx/operators/onnx__OptionalGetElement.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`OptionalHasElement`](https://onnx.ai/onnx/operators/onnx__OptionalHasElement.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`PRelu`](https://onnx.ai/onnx/operators/onnx__PRelu.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`QLinearConv`](https://onnx.ai/onnx/operators/onnx__QLinearConv.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`QLinearMatMul`](https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`QuantizeLinear`](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`RNN`](https://onnx.ai/onnx/operators/onnx__RNN.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`RandomNormal`](https://onnx.ai/onnx/operators/onnx__RandomNormal.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`RandomNormalLike`](https://onnx.ai/onnx/operators/onnx__RandomNormalLike.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`RandomUniform`](https://onnx.ai/onnx/operators/onnx__RandomUniform.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`RandomUniformLike`](https://onnx.ai/onnx/operators/onnx__RandomUniformLike.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Range`](https://onnx.ai/onnx/operators/onnx__Range.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`RegexFullMatch`](https://onnx.ai/onnx/operators/onnx__RegexFullMatch.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`ReverseSequence`](https://onnx.ai/onnx/operators/onnx__ReverseSequence.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`RoiAlign`](https://onnx.ai/onnx/operators/onnx__RoiAlign.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`STFT`](https://onnx.ai/onnx/operators/onnx__STFT.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Scan`](https://onnx.ai/onnx/operators/onnx__Scan.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`SequenceAt`](https://onnx.ai/onnx/operators/onnx__SequenceAt.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`SequenceConstruct`](https://onnx.ai/onnx/operators/onnx__SequenceConstruct.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`SequenceEmpty`](https://onnx.ai/onnx/operators/onnx__SequenceEmpty.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`SequenceErase`](https://onnx.ai/onnx/operators/onnx__SequenceErase.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`SequenceInsert`](https://onnx.ai/onnx/operators/onnx__SequenceInsert.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`SequenceLength`](https://onnx.ai/onnx/operators/onnx__SequenceLength.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`SequenceMap`](https://onnx.ai/onnx/operators/onnx__SequenceMap.html) | ![SUCCESS WITH FALLBACK (via decomposition)](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK%20(via%20decomposition)-FF7700?style=flat&logoColor=white) |
| [`Shape`](https://onnx.ai/onnx/operators/onnx__Shape.html) | ![UNKNOWN (no Node event)](https://img.shields.io/badge/UNKNOWN%20(no%20Node%20event)-DEDEDE?style=flat&logoColor=white) |
| [`Size`](https://onnx.ai/onnx/operators/onnx__Size.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`SoftmaxCrossEntropyLoss`](https://onnx.ai/onnx/operators/onnx__SoftmaxCrossEntropyLoss.html) | ![SUCCESS WITH FALLBACK (via decomposition)](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK%20(via%20decomposition)-FF7700?style=flat&logoColor=white) |
| [`SpaceToDepth`](https://onnx.ai/onnx/operators/onnx__SpaceToDepth.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`SplitToSequence`](https://onnx.ai/onnx/operators/onnx__SplitToSequence.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`StringConcat`](https://onnx.ai/onnx/operators/onnx__StringConcat.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`StringNormalizer`](https://onnx.ai/onnx/operators/onnx__StringNormalizer.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`StringSplit`](https://onnx.ai/onnx/operators/onnx__StringSplit.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`TopK`](https://onnx.ai/onnx/operators/onnx__TopK.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Trilu`](https://onnx.ai/onnx/operators/onnx__Trilu.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |
| [`Unique`](https://onnx.ai/onnx/operators/onnx__Unique.html) | ![SUCCESS WITH FALLBACK](https://img.shields.io/badge/SUCCESS%20WITH%20FALLBACK-FFAA00?style=flat&logoColor=white) |
| [`Where`](https://onnx.ai/onnx/operators/onnx__Where.html) | ![SUCCESS](https://img.shields.io/badge/SUCCESS-00AA44?style=flat&logoColor=white) |

## Global Statistics

- **Total nodes tested:** 152
- **Executable directly (SUCCESS):** 74 (48.7%)
- **Executable via FALLBACK:** 76 (50.0%)
- **UNKNOWN (no Node event):** 2 (1.3%)
- **FAIL:** 0 (0.0%)

### Statistics Pie Chart

![Node Status Distribution](./stats_CUDAExecutionProvider.png)

## README Generation

This file was automatically generated by `report.py` using the `generate_readme` function.

### Related Scripts and Folders

- Generated ONNX models: `models/<provider>/`
- Profiling JSON files: `profiling/<provider>/`
- Test scripts: `main.py`, `utils.py`, `report.py`, `ops/*`

_End of README_
