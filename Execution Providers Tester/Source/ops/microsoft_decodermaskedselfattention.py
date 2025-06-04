# ops/decodermaskedselfattention.py
import numpy as np
import onnx
import onnx.helper
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION

def decodermaskedselfattention_model_builder(op_type, cfg=None):
    batch_size = 2
    input_hidden_size = 16
    num_heads = 2
    head_size = 32
    hidden_size = num_heads * head_size
    v_hidden_size = 64
    past_seq_len = 3
    seq_len = 1
    max_seq_len = past_seq_len + seq_len

    # Inputs
    input_info = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [batch_size, seq_len, input_hidden_size])
    weights_info = onnx.helper.make_tensor_value_info("weights", onnx.TensorProto.FLOAT, [input_hidden_size, hidden_size * 2 + v_hidden_size])
    bias_info = onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [hidden_size * 2 + v_hidden_size])
    past_info = onnx.helper.make_tensor_value_info("past", onnx.TensorProto.FLOAT, [2, batch_size, num_heads, max_seq_len, head_size])
    past_seq_len_info = onnx.helper.make_tensor_value_info("past_sequence_length", onnx.TensorProto.INT32, [1])

    # Outputs
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)
    present_info = onnx.helper.make_tensor_value_info("present", onnx.TensorProto.FLOAT, None)

    node = onnx.helper.make_node(
        "DecoderMaskedSelfAttention",
        inputs=["input", "weights", "bias", "", "past", "", "past_sequence_length"],
        outputs=["output", "present"],
        domain="com.microsoft",
        num_heads=num_heads,
        do_rotary=0,
        mask_filter_value=-10000.0,
        past_present_share_buffer=1  # Important pour CUDA
    )

    graph = onnx.helper.make_graph(
        [node],
        "DecoderMaskedSelfAttentionGraph",
        [input_info, weights_info, bias_info, past_info, past_seq_len_info],
        [output_info, present_info]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("com.microsoft", 1)],
        ir_version=ONNX_RUNTIME_IR_VERSION
    )
    return model

def decodermaskedselfattention_input_generator(session):
    batch_size = 2
    input_hidden_size = 16
    num_heads = 2
    head_size = 32
    hidden_size = num_heads * head_size
    v_hidden_size = 64

    past_seq_len = 3
    seq_len = 1
    max_seq_len = past_seq_len + seq_len

    input_arr = np.random.randn(batch_size, seq_len, input_hidden_size).astype(np.float32)
    weights = np.random.randn(input_hidden_size, hidden_size * 2 + v_hidden_size).astype(np.float32)
    bias = np.random.randn(hidden_size * 2 + v_hidden_size).astype(np.float32)
    past = np.random.randn(2, batch_size, num_heads, max_seq_len, head_size).astype(np.float32)
    past_seq_len = np.array([past_seq_len], dtype=np.int32)

    return {
        "input": input_arr,
        "weights": weights,
        "bias": bias,
        "past": past,
        "past_sequence_length": past_seq_len
    }

SpecialModelBuilders["com.microsoft.DecoderMaskedSelfAttention"] = decodermaskedselfattention_model_builder
SpecialInputGenerators["com.microsoft.DecoderMaskedSelfAttention"] = decodermaskedselfattention_input_generator
