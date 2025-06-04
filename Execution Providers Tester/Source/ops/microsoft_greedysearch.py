# ops/greedysearch.py
import numpy as np
import onnx
import onnx.helper
from onnx import AttributeProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def greedysearch_model_builder(op_type, cfg=None):
    input_ids = onnx.helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT32, [1, 4])
    max_length = onnx.helper.make_tensor_value_info("max_length", onnx.TensorProto.INT32, [1])
    sequences = onnx.helper.make_tensor_value_info("sequences", onnx.TensorProto.INT32, None)

    node = onnx.helper.make_node(
        op_type,
        inputs=["input_ids", "max_length"],
        outputs=["sequences"],
        domain="com.microsoft",
        decoder_start_token_id=50256,
        eos_token_id=50256,
        pad_token_id=0
    )

    graph = onnx.helper.make_graph(
        [node],
        "GreedySearchGraph",
        [input_ids, max_length],
        [sequences]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def greedysearch_input_generator(session):
    input_ids = np.array([[42, 17, 89, 11]], dtype=np.int32)
    max_length = np.array([20], dtype=np.int32)
    return {
        "input_ids": input_ids,
        "max_length": max_length
    }

SpecialModelBuilders["com.microsoft.GreedySearch"] = greedysearch_model_builder
SpecialInputGenerators["com.microsoft.GreedySearch"] = greedysearch_input_generator
