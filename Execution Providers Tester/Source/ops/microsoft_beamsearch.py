# ops/beamsearch.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def beamsearch_model_builder(op_type, cfg=None):
    batch_size = 1
    sequence_length = 5
    vocab_size = 50257
    max_length = 10
    num_beams = 4
    num_return_sequences = 2

    # Entrées du modèle principal
    input_ids_vi = onnx.helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT32, [batch_size, sequence_length])
    max_length_vi = onnx.helper.make_tensor_value_info("max_length", onnx.TensorProto.INT32, [1])
    num_beams_vi = onnx.helper.make_tensor_value_info("num_beams", onnx.TensorProto.INT32, [1])
    num_return_sequences_vi = onnx.helper.make_tensor_value_info("num_return_sequences", onnx.TensorProto.INT32, [1])

    # Sortie du modèle principal
    sequences = onnx.helper.make_tensor_value_info("sequences", onnx.TensorProto.INT32, [batch_size, num_return_sequences, max_length])

    # Sous-graphe Decoder
    decoder_input_ids = onnx.helper.make_tensor_value_info("input_ids", onnx.TensorProto.INT32, [batch_size, 1])
    decoder_past = onnx.helper.make_tensor_value_info("past", onnx.TensorProto.FLOAT, [batch_size, 1, 1])
    decoder_position_ids = onnx.helper.make_tensor_value_info("position_ids", onnx.TensorProto.INT32, [batch_size, 1])
    decoder_attention_mask = onnx.helper.make_tensor_value_info("attention_mask", onnx.TensorProto.INT32, [batch_size, 1])

    decoder_logits = onnx.helper.make_tensor_value_info("logits", onnx.TensorProto.INT32, [batch_size, vocab_size])
    decoder_present = onnx.helper.make_tensor_value_info("present", onnx.TensorProto.FLOAT, [batch_size, 1, 1])

    decoder_node1 = onnx.helper.make_node("Identity", ["input_ids"], ["logits"])
    decoder_node2 = onnx.helper.make_node("Identity", ["past_0"], ["present"])

    decoder_graph = onnx.helper.make_graph(
        [decoder_node1, decoder_node2],
        "DecoderGraph",
        [decoder_input_ids, decoder_position_ids, decoder_attention_mask, decoder_past],
        [decoder_logits, decoder_present]
    )

    # Noeud BeamSearch
    beamsearch_node = onnx.helper.make_node(
        "BeamSearch",
        inputs=[
            "input_ids",        # input_ids
            "max_length",       # max_length
            "",                 # min_length (optionnel)
            "num_beams",        # num_beams
            "num_return_sequences"  # num_return_sequences
        ],
        outputs=["sequences"],
        domain="com.microsoft",
        decoder=decoder_graph,
        eos_token_id=50256,
        pad_token_id=0
    )

    graph = onnx.helper.make_graph(
        [beamsearch_node],
        "BeamSearchGraph",
        [input_ids_vi, max_length_vi, num_beams_vi, num_return_sequences_vi],
        [sequences]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )

    return model

def beamsearch_input_generator(session):
    batch_size = 1
    sequence_length = 5
    vocab_size = 50257
    max_length = 10
    num_beams = 4
    num_return_sequences = 2

    input_ids = np.random.randint(0, vocab_size, size=(batch_size, sequence_length), dtype=np.int32)
    max_length_tensor = np.array([max_length], dtype=np.int32)
    num_beams_tensor = np.array([num_beams], dtype=np.int32)
    num_return_sequences_tensor = np.array([num_return_sequences], dtype=np.int32)

    return {
        "input_ids": input_ids,
        "max_length": max_length_tensor,
        "num_beams": num_beams_tensor,
        "num_return_sequences": num_return_sequences_tensor
    }

SpecialModelBuilders["com.microsoft.BeamSearch"] = beamsearch_model_builder
SpecialInputGenerators["com.microsoft.BeamSearch"] = beamsearch_input_generator
