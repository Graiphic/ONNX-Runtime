# ops/tfidfvectorizer.py
import numpy as np
import onnx
import onnx.helper
from onnx import TensorProto
from utils import SpecialModelBuilders, SpecialInputGenerators, ONNX_RUNTIME_IR_VERSION, ONNX_OPSET_VERSION

def tfidfvectorizer_model_builder(op_type, cfg=None):
    input_tensor = onnx.helper.make_tensor_value_info("X", TensorProto.STRING, [None])
    output_tensor = onnx.helper.make_tensor_value_info("Y", TensorProto.FLOAT, [7])  # 7 ngrams

    pool_strings = ["this", "is", "a", "test", "this is", "is a", "a test"]
    ngram_indexes = list(range(7))
    ngram_counts = [0, 4, 7]  # unigrams start at 0, bigrams start at 4
    weights = [1.0] * 7

    node = onnx.helper.make_node(
        "TfIdfVectorizer",
        inputs=["X"],
        outputs=["Y"],
        mode="TFIDF",
        min_gram_length=1,
        max_gram_length=2,
        max_skip_count=0,
        ngram_counts=ngram_counts,
        ngram_indexes=ngram_indexes,
        pool_strings=pool_strings,
        weights=weights
    )

    graph = onnx.helper.make_graph(
        nodes=[node],
        name="tfidf_graph",
        inputs=[input_tensor],
        outputs=[output_tensor]
    )

    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model

def tfidfvectorizer_input_generator(session):
    # <- crucial : array 1D de tokens, dtype=object
    return {"X": np.array(["this", "is", "a", "test"], dtype=object)}

SpecialModelBuilders["TfIdfVectorizer"] = tfidfvectorizer_model_builder
SpecialInputGenerators["TfIdfVectorizer"] = tfidfvectorizer_input_generator
