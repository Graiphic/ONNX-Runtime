# utils.py

import os
import onnx
import onnx.helper
import onnxruntime as ort
import numpy as np
import pkgutil
import importlib
import ops
import json
from report import generate_report, generate_report_aggregated, generate_readme_split
from datetime import datetime

# Configuration constants
ONNX_RUNTIME_IR_VERSION = 10
ONNX_OPSET_VERSION = 22

SpecialModelBuilders = {}
SpecialInputGenerators = {}


def load_skip_table(path="skip_nodes.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    
def load_not_tested_table(path="untested_nodes.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def map_execution_provider(provider_name):
    mapping = {
        "CPUExecutionProvider": "CPU",
        "CUDAExecutionProvider":       "NVIDIA - CUDA",
        "TensorrtExecutionProvider":   "NVIDIA - TensorRT",
        "OpenVINOExecutionProvider":   "Intel - OpenVINO™",
        "DnnlExecutionProvider":       "Intel - oneDNN",
        "DmlExecutionProvider":        "Windows - DirectML",
        "QNNExecutionProvider":        "Qualcomm - QNN",
        "NnapiExecutionProvider":      "Android - NNAPI",
        "CoreMLExecutionProvider":     "Apple - CoreML",
        "XnnpackExecutionProvider":    "XNNPACK",
        "ROCMExecutionProvider":       "AMD - ROCm",
        "MIGraphXExecutionProvider":   "AMD - MIGraphX",
        "VitisAIExecutionProvider":    "AMD - Vitis AI",
        "AzureExecutionProvider":      "Cloud - Azure",
        "ACLExecutionProvider":        "Arm - ACL",
        "ArmNNExecutionProvider":      "Arm - Arm NN",
        "TVMExecutionProvider":        "Apache - TVM",
        "RknpuExecutionProvider":      "Rockchip - RKNPU",
        "CANNExecutionProvider":       "Huawei - CANN"
    }
    return mapping.get(provider_name, "Unknown")


def default_model_builder(op_type, cfg=None):
    """
    Creates a minimal ONNX model containing a single node of type `op_type`.
    Input is a tensor "X" of shape [1, 1], output is "Y" of unknown shape.
    """
    dtype = onnx.TensorProto.FLOAT
    inp = onnx.helper.make_tensor_value_info("X", dtype, [1, 1])
    out = onnx.helper.make_tensor_value_info("Y", dtype, None)
    node = onnx.helper.make_node(op_type, [inp.name], [out.name])
    graph = onnx.helper.make_graph([node], f"{op_type}_graph", [inp], [out])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_operatorsetid("", ONNX_OPSET_VERSION)]
    )
    model.ir_version = ONNX_RUNTIME_IR_VERSION
    return model


def default_input_generator(session):
    """
    Generates random inputs for each input tensor in the ONNX session.
    If the input type is int64, use random integers; otherwise random floats.
    """
    feed = {}
    for inp in session.get_inputs():
        # Replace any symbolic dimension with 1
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        if inp.type == 'tensor(int64)':
            feed[inp.name] = np.random.randint(0, 10, size=shape, dtype=np.int64)
        else:
            feed[inp.name] = np.random.rand(*shape).astype(np.float32)
    return feed


class OpTest:
    """
    Encapsulates building, saving, and running a single-operator ONNX model.
    """

    def __init__(self, op_type):
        self.op_type = op_type

    def generate_model(self):
        builder = SpecialModelBuilders.get(self.op_type, default_model_builder)
        return builder(self.op_type)

    def generate_input(self, session):
        gen = SpecialInputGenerators.get(self.op_type, default_input_generator)
        return gen(session)

    def save_model(self, model, directory="models"):
        if not os.path.isdir(directory):
            os.makedirs(directory)
        path = os.path.join(directory, f"{self.op_type}.onnx")
        onnx.save(model, path)
        return path


def load_ops(path: str):
    """
    Retourne deux listes triées :
      - basic_ops       : nœuds onnx « classiques »
      - microsoft_ops   : nœuds onnx prefixés par com.microsoft.
    Les lignes vides ou commençant par '#' sont ignorées.
    """
    basic_ops, microsoft_ops = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            (microsoft_ops if name.startswith("com.microsoft.") else basic_ops).append(name)

    return sorted(basic_ops), sorted(microsoft_ops)


def run_tests_and_generate_reports(provider="DnnlExecutionProvider",
                                   test_file="test.txt",
                                   profiling_dir="profiling",
                                   models_dir="models"):

    is_openvino = "OpenVINO" in provider
    if is_openvino:
        try:
            import onnxruntime.tools.add_openvino_win_libs as ov_utils
            ov_utils.add_openvino_libs_to_path()
        except ImportError:
            print(f"Warning: Could not import OpenVINO helpers for provider '{provider}'.")

    if os.path.exists(profiling_dir):
        for f in os.listdir(profiling_dir):
            full_path = os.path.join(profiling_dir, f)
            if os.path.isfile(full_path):
                os.remove(full_path)
    os.makedirs(profiling_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # --- Skip list --- #
    skip_table = load_skip_table()          # charge skip_nodes.json une seule fois
    skip_list = set(skip_table.get("ALL", [])) | set(skip_table.get(provider, []))
    
    # --- Not-tested list --- #
    untested_table = load_not_tested_table()
    untested_list = set(untested_table.get("ALL", [])) | set(untested_table.get(provider, []))

    for _, name, _ in pkgutil.iter_modules(ops.__path__):
        importlib.import_module(f"ops.{name}")

    basic_ops, microsoft_ops = load_ops(test_file)
    ops_to_test = basic_ops + microsoft_ops 
    results = []

    for op in ops_to_test:
        # Known Crash -> FAIL (skipped)
        if op in skip_list:
            results.append((op, provider, None, "FAIL (skipped: known crash)"))
            continue
        # Unfinished Node -> NOT TESTED
        if op in untested_list:
            results.append((op, provider, None,
                            "NOT TESTED (model unavailable)"))
            continue
            
        tester = OpTest(op)

        try:
            model = tester.generate_model()
            tester.save_model(model, directory=models_dir)
        except Exception as e:
            results.append((op, provider, None, f"FAIL generate_model: {e}"))
            continue

        opts = ort.SessionOptions()
        opts.enable_profiling = True
        opts.profile_file_prefix = os.path.join(profiling_dir, f"profile_{op}")
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.log_severity_level = 4

        skip_optimized = provider in [
            "OpenVINOExecutionProvider",
            "TensorrtExecutionProvider",
            "DnnlExecutionProvider"
        ]
        if not skip_optimized:
            opts.optimized_model_filepath = os.path.join(models_dir, f"{op}_optimized.onnx")

        try:
            if is_openvino:
                sess = ort.InferenceSession(
                    model.SerializeToString(),
                    sess_options=opts,
                    providers=[provider],
                    provider_options=[{"device_type": "CPU"}]
                )
            elif provider == "TensorrtExecutionProvider":
                sess = ort.InferenceSession(
                    model.SerializeToString(),
                    sess_options=opts,
                    providers=[
                        "TensorrtExecutionProvider",
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider"
                    ]
                )
            else:
                sess = ort.InferenceSession(
                    model.SerializeToString(),
                    sess_options=opts,
                    providers=[provider]
                )
        except Exception as e:
            results.append((op, provider, None, f"FAIL session creation: {e}"))
            continue

        try:
            feed = tester.generate_input(sess)
        except Exception as e:
            results.append((op, provider, None, f"FAIL generate_input: {e}"))
            continue

        try:
            iobinding_cb = feed.pop("__iobinding__", None)
            if iobinding_cb:
                iobinding_cb(sess)
            else:
                sess.run(None, feed)
            profile_path = sess.end_profiling()
        except Exception as e:
            results.append((op, provider, None, f"FAIL run: {e}"))
            continue

        try:
            with open(profile_path, 'r') as f:
                root = json.load(f)

            events = root["traceEvents"] if isinstance(root, dict) and "traceEvents" in root else root
            op_events = [e for e in events if e.get("cat") == "Node" and e.get("args", {}).get("op_name") == op]

            if op_events:
                used = op_events[0]["args"]["provider"]
                status = "SUCCESS" if used == provider else "SUCCESS WITH FALLBACK"
            else:
                subops = [
                    e for e in events
                    if e.get("cat") == "Node" and e.get("args", {}).get("op_name") not in (
                        "MemcpyFromHost", "MemcpyToHost"
                    )
                ]
                if subops:
                    sub_providers = [e["args"]["provider"] for e in subops]
                    if all(p == provider for p in sub_providers):
                        used = provider
                        status = "SUCCESS (via decomposition)"
                    else:
                        used = ",".join(sorted(set(sub_providers)))
                        status = "SUCCESS WITH FALLBACK (via decomposition)"
                else:
                    used = None
                    status = "UNKNOWN (no Node event)"

            results.append((op, provider, used, status))
        except Exception as e:
            results.append((op, provider, None, f"FAIL parsing JSON: {e}"))
            continue
        
        
        # OpenVINO fallback re-test with complexified model
        # Fallback re-test with complexified model for OpenVINO and TensorRT
        is_tensorrt = provider == "TensorrtExecutionProvider"
        if (is_openvino or is_tensorrt) and status.startswith("SUCCESS WITH FALLBACK"):
            try:
                # ⚠️ Utilise le modèle de base pour générer les bons inputs
                feed2 = tester.generate_input(sess)
        
                if isinstance(feed2, tuple):
                    feed2 = feed2[0]
                if not isinstance(feed2, dict):
                    raise TypeError(f"generate_input for {op} must return a dict, got {type(feed2)}")
        
                # Complexifie ensuite le modèle
                complex_model = complexify_model_for_openvino(model)
                complex_path = os.path.join(models_dir, f"{op}_complex.onnx")
                onnx.save(complex_model, complex_path)
                opts.enable_profiling = True
                opts.profile_file_prefix = os.path.join(profiling_dir, f"profile_{op}_complex")
        
                if is_openvino:
                    sess2 = ort.InferenceSession(
                        complex_model.SerializeToString(),
                        sess_options=opts,
                        providers=[provider],
                        provider_options=[{"device_type": "CPU"}]
                    )
                elif is_tensorrt:
                    sess2 = ort.InferenceSession(
                        complex_model.SerializeToString(),
                        sess_options=opts,
                        providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
                    )
        
                # Ajoute `noise` manuellement
                if "noise" in [i.name for i in sess2.get_inputs()]:
                    ref_input = next(iter(feed2.values()))
                    feed2["noise"] = np.array(1, dtype=ref_input.dtype)
        
                sess2.run(None, feed2)
                profile_path2 = sess2.end_profiling()
        
                with open(profile_path2, 'r') as f:
                    root2 = json.load(f)
        
                events2 = root2["traceEvents"] if isinstance(root2, dict) else root2
        
                status2 = executed_on_provider_strict(op, events2, provider)
                results[-1] = (op, provider, provider, status2)


        
            except Exception as e:
                print(f"[OpenVINO fallback check] Failed for {op}: {e}")


    

    map_name = map_execution_provider(provider)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_output_dir = os.path.join(project_root, map_name)
    os.makedirs(base_output_dir, exist_ok=True)
    #print(results)
    generate_report(results, provider, base_output_dir, models_dir)
    generate_report_aggregated(results, provider, base_output_dir, models_dir)
    #generate_readme(results, provider, base_output_dir)
    generate_readme_split(results, provider, base_output_dir)



def run_all_providers_and_generate_reports(test_file="test.txt",
                                           base_profiling_dir="profiling",
                                           base_models_dir="models"):
    """
    Detects all available ONNX Runtime execution providers, and for each provider:
      - Creates subdirectories base_profiling_dir/<provider> and base_models_dir/<provider>
      - Calls run_tests_and_generate_reports(provider, test_file, ...)
    """
    providers = ort.get_available_providers()
    if not providers:
        print("No ONNXRuntime execution providers found.")
        return

    for provider in providers:
        profiling_dir = os.path.join(base_profiling_dir, provider)
        models_dir = os.path.join(base_models_dir, provider)
        print(f"\n=== Running tests with provider: {provider} ===")
        run_tests_and_generate_reports(
            provider=provider,
            test_file=test_file,
            profiling_dir=profiling_dir,
            models_dir=models_dir
        )



def complexify_model_for_openvino(model: onnx.ModelProto) -> onnx.ModelProto:
    from onnx import helper, TensorProto

    graph = model.graph
    input0 = graph.input[0]
    dtype = input0.type.tensor_type.elem_type
    input_name = input0.name

    # Crée une nouvelle entrée scalaire (noise)
    new_input_name = "noise"
    new_input = helper.make_tensor_value_info(new_input_name, dtype, [])
    graph.input.append(new_input)

    # Mul pour numériques, And pour booléens
    numeric_types = {
        TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE,
        TensorProto.INT8, TensorProto.INT16, TensorProto.INT32, TensorProto.INT64,
        TensorProto.UINT8, TensorProto.UINT16, TensorProto.UINT32, TensorProto.UINT64
    }
    op_type = "Mul" if dtype in numeric_types else "And"

    new_nodes = []
    prev_output = None
    for i in range(10):
        in0 = input_name if i == 0 else prev_output
        out_name = f"{op_type.lower()}_{i}"
        node = helper.make_node(op_type, inputs=[in0, new_input_name], outputs=[out_name])
        new_nodes.append(node)
        prev_output = out_name

    # Redirige les entrées d’origine
    for node in graph.node:
        for idx, name in enumerate(node.input):
            if name == input_name:
                node.input[idx] = prev_output

    for node in reversed(new_nodes):
        graph.node.insert(0, node)

    return model


def executed_on_provider_strict(op_name: str, events: list, provider: str) -> str:
    """
    Vérifie si TOUTES les exécutions sont sur le provider (ex: OpenVINO).
    Si un kernel tourne sur un autre EP, c'est un fallback.
    """
    nodes = [e for e in events if e.get("cat") == "Node"]
    fallback_eps = {
        e["args"]["provider"]
        for e in nodes
        if e["args"].get("provider") != provider
        and e["args"].get("op_name") != "MemcpyFromHost"
        and e["args"].get("op_name") != "MemcpyToHost"
    }

    if fallback_eps:
        return "SUCCESS WITH FALLBACK"

    # Si le nœud `op_name` est explicitement présent (exécution directe)
    if any(e["args"].get("op_name") == op_name and e["args"]["provider"] == provider for e in nodes):
        return "SUCCESS"

    # Sinon, on considère que c'est une complexification réussie
    return "SUCCESS (with complexification)"
