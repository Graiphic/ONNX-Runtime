# ops/microsoft_mulinteger.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def mulinteger_model_builder(op_type, cfg=None):
    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.UINT8, [2, 3])
    B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.UINT8, [3])
    C = onnx.helper.make_tensor_value_info("C", onnx.TensorProto.INT32, [2, 3])

    node = onnx.helper.make_node(
        "MulInteger",
        inputs=["A", "", "B", ""],  # A_zero_point, B_zero_point non utilisés
        outputs=["C"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "MulIntegerGraph",
        [A, B],
        [C]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def mulinteger_input_generator(session):
    A = np.random.randint(0, 255, size=(2, 3), dtype=np.uint8)
    B = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
    return {"A": A, "B": B}

SpecialModelBuilders["com.microsoft.MulInteger"] = mulinteger_model_builder
SpecialInputGenerators["com.microsoft.MulInteger"] = mulinteger_input_generator
