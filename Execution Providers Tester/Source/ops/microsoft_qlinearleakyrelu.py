# ops/microsoft_qlinearleakyrelu.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def qlinearleakyrelu_model_builder(op_type, cfg=None):
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.UINT8, [2, 3])
    X_scale = onnx.helper.make_tensor_value_info("X_scale", onnx.TensorProto.FLOAT, [])
    X_zp = onnx.helper.make_tensor_value_info("X_zero_point", onnx.TensorProto.UINT8, [])
    Y_scale = onnx.helper.make_tensor_value_info("Y_scale", onnx.TensorProto.FLOAT, [])
    Y_zp = onnx.helper.make_tensor_value_info("Y_zero_point", onnx.TensorProto.UINT8, [])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.UINT8, None)

    node = onnx.helper.make_node(
        "QLinearLeakyRelu",
        inputs=["X", "X_scale", "X_zero_point", "Y_scale", "Y_zero_point"],
        outputs=["Y"],
        domain="com.microsoft",
        alpha=0.01
    )

    graph = onnx.helper.make_graph(
        [node],
        "QLinearLeakyReluGraph",
        [X, X_scale, X_zp, Y_scale, Y_zp],
        [Y]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def qlinearleakyrelu_input_generator(session):
    inputs = {
        "X": np.random.randint(0, 256, size=(2, 3), dtype=np.uint8),
        "X_scale": np.array(0.02, dtype=np.float32),
        "X_zero_point": np.array(128, dtype=np.uint8),
        "Y_scale": np.array(0.02, dtype=np.float32),
        "Y_zero_point": np.array(128, dtype=np.uint8)
    }
    return inputs

SpecialModelBuilders["com.microsoft.QLinearLeakyRelu"] = qlinearleakyrelu_model_builder
SpecialInputGenerators["com.microsoft.QLinearLeakyRelu"] = qlinearleakyrelu_input_generator
