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
from report import generate_report, generate_report_aggregated, generate_readme
from datetime import datetime

# Configuration constants
ONNX_RUNTIME_IR_VERSION = 10
ONNX_OPSET_VERSION = 22

SpecialModelBuilders = {}
SpecialInputGenerators = {}

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


def load_ops(path):
    """
    Reads `path` line by line and returns a list of op names,
    ignoring empty lines and lines starting with '#'.
    """
    with open(path, 'r') as f:
        return [l.strip() for l in f if l.strip() and not l.startswith('#')]


def run_tests_and_generate_reports(provider="DnnlExecutionProvider",
                                   test_file="test.txt",
                                   profiling_dir="profiling",
                                   models_dir="models"):
    """
    Executes all ops listed in `test_file` under the specified ONNX execution provider.
    Collects results, then calls the report generation functions.

    - provider: the ONNX Execution Provider to test (string).
    - test_file: path to a text file listing ops, one per line.
    - profiling_dir: directory where profiling JSONs will be saved.
    - models_dir: directory where generated ONNX models will be stored.

    If provider is "TensorrtExecutionProvider", it will attempt to use TensorRT first,
    then CUDAExecutionProvider as fallback, then CPUExecutionProvider if neither is available.
    Automatically configures optimized model saving for providers that support it,
    skips optimized saving for OpenVINO, TensorRT, and Dnnl providers,
    and applies OpenVINO-specific setup if provider contains "OpenVINO".
    """
    is_openvino = "OpenVINO" in provider
    if is_openvino:
        try:
            import onnxruntime.tools.add_openvino_win_libs as ov_utils
            ov_utils.add_openvino_libs_to_path()
        except ImportError:
            print(f"Warning: Could not import OpenVINO helpers for provider '{provider}'.")

    # --- Clean/Create directories ---
    if os.path.exists(profiling_dir):
        for f in os.listdir(profiling_dir):
            full_path = os.path.join(profiling_dir, f)
            if os.path.isfile(full_path):
                os.remove(full_path)
    os.makedirs(profiling_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # --- Dynamically import all modules under ops.* ---
    for _, name, _ in pkgutil.iter_modules(ops.__path__):
        importlib.import_module(f"ops.{name}")

    # --- Load the list of ops to test ---
    ops_to_test = load_ops(test_file)

    # Each element: (op_name, provider, used_provider, status)
    results = []

    for op in ops_to_test:
        tester = OpTest(op)

        # 1) Generate and save the ONNX model for this op
        try:
            model = tester.generate_model()
            tester.save_model(model, directory=models_dir)
        except Exception as e:
            results.append((op, provider, None, f"FAIL generate_model: {e}"))
            continue

        # 2) Configure SessionOptions for profiling
        opts = ort.SessionOptions()
        opts.enable_profiling = True
        opts.profile_file_prefix = os.path.join(profiling_dir, f"profile_{op}")
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.log_severity_level = 3

        # 2a) If provider supports optimized output, set optimized_model_filepath
        # Skip for OpenVINO, TensorRT, Dnnl
        skip_optimized = provider in [
            "OpenVINOExecutionProvider",
            "TensorrtExecutionProvider",
            "DnnlExecutionProvider"
        ]
        if not skip_optimized:
            opts.optimized_model_filepath = os.path.join(models_dir, f"{op}_optimized.onnx")

        # 3) Create the inference session
        try:
            if is_openvino:
                sess = ort.InferenceSession(
                    model.SerializeToString(),
                    sess_options=opts,
                    providers=[provider],
                    provider_options=[{"device_type": "CPU"}]
                )
            elif provider == "TensorrtExecutionProvider":
                # Use TensorRT first, then CUDA, then CPU
                preferred_providers = [
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider"
                ]
                sess = ort.InferenceSession(
                    model.SerializeToString(),
                    sess_options=opts,
                    providers=preferred_providers
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

        # 4) Generate the input tensor(s)
        try:
            feed = tester.generate_input(sess)
        except Exception as e:
            results.append((op, provider, None, f"FAIL generate_input: {e}"))
            continue

        # 5) Run inference and retrieve the profiling JSON path
        try:
            sess.run(None, feed)
            profile_path = sess.end_profiling()
        except Exception as e:
            results.append((op, provider, None, f"FAIL run: {e}"))
            continue

        # 6) Parse the profiling JSON to determine which provider was used
        try:
            with open(profile_path, 'r') as f:
                root = json.load(f)

            # Normalize events list
            if isinstance(root, dict) and "traceEvents" in root:
                events = root["traceEvents"]
            elif isinstance(root, list):
                events = root
            else:
                raise RuntimeError(f"Unexpected profiling format: {type(root)}")

            # 6a) Direct Node events for the op
            op_events = [
                e for e in events
                if e.get("cat") == "Node"
                   and e.get("args", {}).get("op_name") == op
            ]

            if op_events:
                used = op_events[0]["args"]["provider"]
                status = "SUCCESS" if used == provider else "SUCCESS WITH FALLBACK"
            else:
                # 6b) Decomposition case: collect all sub-ops (excluding Memcpy)
                subops = [
                    e for e in events
                    if e.get("cat") == "Node"
                       and e.get("args", {}).get("op_name") not in (
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

    # --- Generate the README in Markdown (with pie chart) ---
    map_name     = map_execution_provider(provider)  # e.g. "NVIDIA - CUDA", "AMD - ROCm", etc.
    # Always navigate to the project root by looking at the folder of this file
    project_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_output_dir = os.path.join(project_root, map_name)
    os.makedirs(base_output_dir, exist_ok=True)

    # --- Generate the detailed and aggregated Excel reports ---
    generate_report(results, provider, base_output_dir, models_dir)
    generate_report_aggregated(results, provider, base_output_dir, models_dir)
    generate_readme(results, provider, base_output_dir)


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
