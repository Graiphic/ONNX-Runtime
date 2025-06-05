# ops/microsoft_nhwcfusedconv.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def nhwcfusedconv_model_builder(op_type, cfg=None):
    N, H, Wd, C_in, C_out, k = 1, 5, 5, 3, 4, 3

    X_shape = [int(N), int(H), int(Wd), int(C_in)]
    W_shape = [int(C_out), int(C_in), int(k), int(k)]
    Y_shape = [int(N), int(H), int(Wd), int(C_out)]

    X = onnx.helper.make_tensor_value_info("X", TensorProto.FLOAT16, X_shape)
    W = onnx.helper.make_tensor_value_info("W", TensorProto.FLOAT16, W_shape)
    Y = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT16, Y_shape)

    node = onnx.helper.make_node(
        "NhwcFusedConv",
        inputs=["X", "W"],
        outputs=["Y"],
        domain="com.microsoft",
        kernel_shape=[k, k],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        group=1
    )

    graph = onnx.helper.make_graph(
        [node],
        "NhwcFusedConvGraph",
        [X, W],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_operatorsetid("com.microsoft", 1)
        ],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model


def nhwcfusedconv_input_generator(session):
    N, H, W, C_in, C_out, k = 1, 5, 5, 3, 4, 3
    X = np.random.randn(N, H, W, C_in).astype(np.float16)
    W = np.random.randn(C_out, C_in, k, k).astype(np.float16)
    return {"X": X, "W": W}

SpecialModelBuilders["com.microsoft.NhwcFusedConv"] = nhwcfusedconv_model_builder
SpecialInputGenerators["com.microsoft.NhwcFusedConv"] = nhwcfusedconv_input_generator
