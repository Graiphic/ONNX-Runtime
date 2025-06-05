# ops/greedysearch.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def greedysearch_model_builder(op_type, cfg=None):
    # Entrées principales du nœud GreedySearch
    input_ids  = onnx.helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT32, [1, 5])
    max_length = onnx.helper.make_tensor_value_info("max_length", onnx.TensorProto.INT32, [1])
    sequences  = onnx.helper.make_tensor_value_info("sequences", onnx.TensorProto.INT32, None)

    # Sous-graphe decoder : 4 entrées, 2 sorties
    decoder_inputs = [
        onnx.helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT32, [1, 1]),
        onnx.helper.make_tensor_value_info("position_ids", onnx.TensorProto.INT32, [1, 1]),
        onnx.helper.make_tensor_value_info("attention_mask", onnx.TensorProto.INT32, [1, 1]),
        onnx.helper.make_tensor_value_info("past_state", onnx.TensorProto.FLOAT, [1, 2]),
    ]
    decoder_outputs = [
        onnx.helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, 10]),
        onnx.helper.make_tensor_value_info("new_past_state", onnx.TensorProto.FLOAT, [1, 2]),
    ]
    nodes = [
        onnx.helper.make_node("Identity", inputs=["past_state"], outputs=["logits"]),
        onnx.helper.make_node("Identity", inputs=["past_state"], outputs=["new_past_state"]),
    ]
    decoder_graph = onnx.helper.make_graph(nodes, "decoder_graph", decoder_inputs, decoder_outputs)

    node = onnx.helper.make_node(
        "GreedySearch",
        inputs=["input_ids", "max_length"],
        outputs=["sequences"],
        domain="com.microsoft",
        decoder=decoder_graph,
        decoder_start_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        model_type=0
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
    feed = {}
    for inp in session.get_inputs():
        if inp.name == "input_ids":
            feed[inp.name] = np.array([[10, 20, 30, 40, 50]], dtype=np.int32)
        elif inp.name == "max_length":
            feed[inp.name] = np.array([10], dtype=np.int32)
    return feed

SpecialModelBuilders["com.microsoft.GreedySearch"] = greedysearch_model_builder
SpecialInputGenerators["com.microsoft.GreedySearch"] = greedysearch_input_generator
