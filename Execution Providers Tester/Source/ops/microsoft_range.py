# ops/microsoft_range.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def range_model_builder(op_type, cfg=None):
    dtype = onnx.TensorProto.INT32

    # Déclarations (scalaires ou [1])
    start = onnx.helper.make_tensor("start", dtype, [], [2])
    limit = onnx.helper.make_tensor("limit", dtype, [], [10])
    delta = onnx.helper.make_tensor("delta", dtype, [], [2])

    y = onnx.helper.make_tensor_value_info("Y", dtype, None)

    node = onnx.helper.make_node(
        "Range",
        inputs=["start", "limit", "delta"],
        outputs=["Y"],
        domain="com.microsoft"
    )

    graph = onnx.helper.make_graph(
        [node],
        "RangeGraph",
        [],
        [y],
        initializer=[start, limit, delta]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def range_input_generator(session):
    return {}

SpecialModelBuilders["com.microsoft.Range"] = range_model_builder
SpecialInputGenerators["com.microsoft.Range"] = range_input_generator
