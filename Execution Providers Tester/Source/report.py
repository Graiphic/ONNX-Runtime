# report.py

import os
import re
import subprocess
from datetime import datetime
import openpyxl
from openpyxl.styles import PatternFill, Border, Side
from openpyxl.chart import Reference
from openpyxl.chart.pie_chart import PieChart
from openpyxl.chart.series import DataPoint, GraphicalProperties
from openpyxl.chart.label import DataLabelList
from openpyxl.drawing.colors import ColorChoice
from openpyxl.utils import get_column_letter
from openpyxl.chart.layout import Layout, ManualLayout
from collections import Counter
import onnx
import onnxruntime as ort
import platform
import urllib.parse
from collections import defaultdict


try:
    import cpuinfo  # To get a more readable CPU name
except ImportError:
    cpuinfo = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------
# Excel Report Generation: Detailed (non-aggregated)
# --------------------------------------------------

def generate_report(results, provider, profiling_dir, models_dir):
    """
    Generates an Excel file report_{provider}.xlsx from the `results` list.
    Status categories handled:
      - SUCCESS
      - SUCCESS (via decomposition)
      - SUCCESS WITH FALLBACK
      - SUCCESS WITH FALLBACK (via decomposition)
      - UNKNOWN (no Node event)
      - NOT TESTED
      - SKIPPED
      - FAIL
    """

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Report"
    ws.append(["op", "provider", "used_provider", "status"])

    color_map = {
        "SUCCESS": "00AA44",
        "SUCCESS (via decomposition)": "007700",
        "SUCCESS WITH FALLBACK": "FFAA00",
        "SUCCESS WITH FALLBACK (via decomposition)": "FF7700",
        "UNKNOWN (no Node event)": "DEDEDE",
        "NOT TESTED": "4D7CFE",
        "SKIPPED": "CCCCCC",
        "FAIL": "FF0000",
    }

    def get_fill_for_status(status):
        for key in color_map:
            if status == key or status.startswith(key):
                return PatternFill("solid", fgColor=color_map[key])
        return PatternFill("solid", fgColor=color_map["FAIL"])

    for row in results:
        op_name, provider_used, used_provider, status = row
        display_status = "SUCCESS" if status.startswith("SUCCESS (with complexification)") else status
        ws.append([op_name, provider_used, used_provider or "", display_status])
        ws.cell(row=ws.max_row, column=4).fill = get_fill_for_status(display_status)

    for col_idx in range(1, 5):
        max_len = max(len(str(cell.value)) for cell in ws[get_column_letter(col_idx)] if cell.value)
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 80)

    border = Border(left=Side(style="thin"), right=Side(style="thin"),
                    top=Side(style="thin"), bottom=Side(style="thin"))
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=4):
        for cell in row:
            cell.border = border

    counter = Counter()
    for _, _, _, status in results:
        normalized = "SUCCESS" if status.startswith("SUCCESS (with complexification)") else status
        matched = False
        for key in color_map:
            if normalized == key or normalized.startswith(key):
                counter[key] += 1
                matched = True
                break
        if not matched:
            counter["FAIL"] += 1

    ws_data = wb.create_sheet("Data_PieChart")
    ws_data.append(["Category", "Count", "Percentage"])
    total = sum(counter.values()) or 1

    ordered_keys = list(color_map.keys())
    for key in ordered_keys:
        count = counter.get(key, 0)
        percent = round(100 * count / total, 1)
        ws_data.append([key, count, percent])

    ws_data.cell(row=ws_data.max_row + 1, column=1, value="Report generated on")
    ws_data.cell(row=ws_data.max_row, column=2, value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    for col_idx in range(1, 4):
        max_len = max(len(str(cell.value)) for cell in ws_data[get_column_letter(col_idx)] if cell.value)
        ws_data.column_dimensions[get_column_letter(col_idx)].width = max_len + 2

    chart = PieChart()
    chart.title = "Status Distribution"
    chart.width = 20
    chart.height = 12
    chart.legend.position = 'r'

    cats = Reference(ws_data, min_col=1, min_row=2, max_row=9)
    vals = Reference(ws_data, min_col=2, min_row=2, max_row=9)
    chart.add_data(vals, titles_from_data=False)
    chart.set_categories(cats)

    series = chart.series[0]
    for idx, key in enumerate(ordered_keys):
        if key in color_map:
            dp = DataPoint(idx=idx, spPr=GraphicalProperties(solidFill=ColorChoice(srgbClr=color_map[key])))
            series.dPt.append(dp)

    chart.dataLabels = DataLabelList()
    chart.dataLabels.showPercent = True
    chart.plot_area.layout = Layout(manualLayout=ManualLayout(x=0.00, y=0.20, w=0.50, h=0.60))
    ws.add_chart(chart, "F2")

    report_path = os.path.join(profiling_dir, f"report_{provider}.xlsx")
    wb.save(report_path)
    print("Report generated:", report_path)
    print(f" • Profiling JSON files are in '{profiling_dir}/'")
    print(f" • Optimized models are in '{models_dir}/'")


# --------------------------------------------------
# Excel Report Generation: Aggregated
# --------------------------------------------------

def generate_report_aggregated(results, provider, profiling_dir, models_dir):
    """
    Generates an Excel file report_{provider}_aggregated.xlsx from the `results` list.
    Aggregated categories:
      - SUCCESS
      - SUCCESS WITH FALLBACK
      - UNKNOWN (no Node event)
      - NOT TESTED
      - SKIPPED
      - FAIL
    """

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Report"
    ws.append(["op", "provider", "used_provider", "status"])

    agg_color_map = {
        "SUCCESS": "00AA44",
        "SUCCESS WITH FALLBACK": "FFAA00",
        "UNKNOWN (no Node event)": "DEDEDE",
        "NOT TESTED": "4D7CFE",
        "SKIPPED": "CCCCCC",
        "FAIL": "FF0000",
    }

    def get_fill_for_agg_status(status):
        normalized = "SUCCESS" if status.startswith("SUCCESS (with complexification)") else status
        if normalized.startswith("SUCCESS") and "FALLBACK" not in normalized:
            return PatternFill("solid", fgColor=agg_color_map["SUCCESS"])
        elif "WITH FALLBACK" in normalized:
            return PatternFill("solid", fgColor=agg_color_map["SUCCESS WITH FALLBACK"])
        elif normalized.startswith("UNKNOWN"):
            return PatternFill("solid", fgColor=agg_color_map["UNKNOWN (no Node event)"])
        elif normalized.startswith("NOT TESTED"):
            return PatternFill("solid", fgColor=agg_color_map["NOT TESTED"])
        elif normalized.startswith("SKIPPED"):
            return PatternFill("solid", fgColor=agg_color_map["SKIPPED"])
        elif normalized.startswith("FAIL"):
            return PatternFill("solid", fgColor=agg_color_map["FAIL"])
        return PatternFill("solid", fgColor=agg_color_map["FAIL"])

    for row in results:
        op_name, provider_used, used_provider, status = row
        display_status = "SUCCESS" if status.startswith("SUCCESS (with complexification)") else status
        ws.append([op_name, provider_used, used_provider or "", display_status])
        ws.cell(row=ws.max_row, column=4).fill = get_fill_for_agg_status(display_status)

    for col_idx in range(1, 5):
        max_len = max(len(str(cell.value)) for cell in ws[get_column_letter(col_idx)] if cell.value)
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 80)

    border = Border(left=Side(style="thin"), right=Side(style="thin"),
                    top=Side(style="thin"), bottom=Side(style="thin"))
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=4):
        for cell in row:
            cell.border = border

    counter_agg = Counter()
    for _, _, _, status in results:
        normalized = "SUCCESS" if status.startswith("SUCCESS (with complexification)") else status
        if normalized.startswith("SUCCESS") and "FALLBACK" not in normalized:
            counter_agg["SUCCESS"] += 1
        elif "WITH FALLBACK" in normalized:
            counter_agg["SUCCESS WITH FALLBACK"] += 1
        elif normalized.startswith("UNKNOWN"):
            counter_agg["UNKNOWN (no Node event)"] += 1
        elif normalized.startswith("NOT TESTED"):
            counter_agg["NOT TESTED"] += 1
        elif normalized.startswith("SKIPPED"):
            counter_agg["SKIPPED"] += 1
        elif normalized.startswith("FAIL"):
            counter_agg["FAIL"] += 1
        else:
            counter_agg["FAIL"] += 1

    ws_data = wb.create_sheet("Data_PieChart")
    ws_data.append(["Category", "Count", "Percentage"])

    ordered_keys = list(agg_color_map.keys())
    total_agg = sum(counter_agg.values()) or 1
    for key in ordered_keys:
        count = counter_agg.get(key, 0)
        percent = round(100 * count / total_agg, 1)
        ws_data.append([key, count, percent])

    ts_row = ws_data.max_row + 1
    ws_data.cell(row=ts_row, column=1, value="Report generated on")
    ws_data.cell(row=ts_row, column=2, value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    for col_idx in range(1, 4):
        max_len = max(len(str(cell.value)) for cell in ws_data[get_column_letter(col_idx)] if cell.value)
        ws_data.column_dimensions[get_column_letter(col_idx)].width = max_len + 2

    chart = PieChart()
    chart.title = "Status Distribution (aggregated)"
    chart.width = 20
    chart.height = 12
    chart.legend.position = 'r'

    cats = Reference(ws_data, min_col=1, min_row=2, max_row=7)
    vals = Reference(ws_data, min_col=2, min_row=2, max_row=7)
    chart.add_data(vals, titles_from_data=False)
    chart.set_categories(cats)

    series = chart.series[0]
    for idx, key in enumerate(ordered_keys):
        if key in agg_color_map:
            dp = DataPoint(idx=idx, spPr=GraphicalProperties(solidFill=ColorChoice(srgbClr=agg_color_map[key])))
            series.dPt.append(dp)

    chart.dataLabels = DataLabelList()
    chart.dataLabels.showPercent = True
    chart.plot_area.layout = Layout(manualLayout=ManualLayout(x=0.00, y=0.20, w=0.50, h=0.60))
    ws.add_chart(chart, "F2")

    report_path = os.path.join(profiling_dir, f"report_{provider}_aggregated.xlsx")
    wb.save(report_path)

    print("Aggregated report generated:", report_path)
    print(f" • Profiling JSON files are in '{profiling_dir}/'")
    print(f" • Optimized models are in '{models_dir}/'")



# --------------------------------------------------
# Utility functions to retrieve cuDNN and TensorRT versions
# --------------------------------------------------

def parse_cudnn_header(header_path):
    """
    Reads the cudnn_version.h or cudnn.h file and extracts CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL.
    Returns "X.Y.Z" if found, otherwise None.
    """
    try:
        with open(header_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return None

    major = re.search(r"#define\s+CUDNN_MAJOR\s+(\d+)", content)
    minor = re.search(r"#define\s+CUDNN_MINOR\s+(\d+)", content)
    patch = re.search(r"#define\s+CUDNN_PATCHLEVEL\s+(\d+)", content)
    if major and minor and patch:
        return f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}"
    return None


def get_cudnn_version():
    """
    Searches multiple possible locations to find cudnn_version.h or cudnn.h under the CUDA toolkit
    and returns the version formatted as "X.Y.Z" if found.
    """
    cuda_candidates = []

    # 1) If CUDA_PATH is set, add it
    cuda_env = os.environ.get("CUDA_PATH")
    if cuda_env:
        cuda_candidates.append(cuda_env)
        # If CUDA_PATH ends with "vX.Y", also check the parent directory
        if os.path.basename(cuda_env).lower().startswith("v"):
            cuda_candidates.append(os.path.dirname(cuda_env))

    # 2) Standard installation path under Program Files on Windows
    prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    default_root = os.path.join(prog_files, "NVIDIA GPU Computing Toolkit", "CUDA")
    cuda_candidates.append(default_root)

    # 3) Iterate through each candidate directory to detect cudnn_version.h or cudnn.h
    for base in cuda_candidates:
        if not os.path.isdir(base):
            continue

        # 3.a) Check if base/include contains cudnn_version.h or cudnn.h
        include_dir = os.path.join(base, "include")
        if os.path.isdir(include_dir):
            for filename in ("cudnn_version.h", "cudnn.h"):
                path_h = os.path.join(include_dir, filename)
                if os.path.isfile(path_h):
                    version = parse_cudnn_header(path_h)
                    if version:
                        return version

        # 3.b) Otherwise, search each subdirectory (e.g., v11.8, v12.5, etc.)
        for sub in os.listdir(base):
            sub_include = os.path.join(base, sub, "include")
            if not os.path.isdir(sub_include):
                continue
            for filename in ("cudnn_version.h", "cudnn.h"):
                path_h = os.path.join(sub_include, filename)
                if os.path.isfile(path_h):
                    version = parse_cudnn_header(path_h)
                    if version:
                        return version

    return "Unknown"


def get_tensorrt_version():
    """
    Attempts to call 'trtexec' to retrieve the TensorRT version.
    - If 'trtexec' prints a string like 'TensorRT.trtexec [TensorRT v100900]',
      extract '100900' and convert it to '10.9.0'.
    - If 'trtexec --version' already returns 'TensorRT v10.9.0', return '10.9.0' directly.
    - On error or if nothing is found, return 'Unknown'.
    """
    # Allow either a dot or a space between "TensorRT" and "trtexec"
    version_pattern = re.compile(r"TensorRT[.\s]+trtexec\s+\[TensorRT\s+v([\d\.]+)\]", re.IGNORECASE)

    def normalize(raw: str) -> str:
        # If the string already contains dots, return it as-is
        if "." in raw:
            return raw
        # If it's a 6-digit number like "100900", split into major, minor, patch
        if raw.isdigit() and len(raw) == 6:
            major = int(raw[0:2])
            minor = int(raw[2:4])
            patch = int(raw[4:6])
            return f"{major}.{minor}.{patch}"
        return raw

    try:
        # 1) Call "trtexec" without check=True to allow non-zero return codes
        proc = subprocess.run(
            ["trtexec"],
            capture_output=True,
            text=True
        )
        output = proc.stdout + proc.stderr

        match = version_pattern.search(output)
        if match:
            return normalize(match.group(1))

        # 2) If not found, try with "--version"
        proc2 = subprocess.run(
            ["trtexec", "--version"],
            capture_output=True,
            text=True
        )
        output2 = proc2.stdout + proc2.stderr
        match2 = version_pattern.search(output2)
        if match2:
            return normalize(match2.group(1))

    except FileNotFoundError:
        return "Unknown"
    except Exception:
        return "Unknown"

    return "Unknown"


# --------------------------------------------------
# Main function to generate the README in English
# --------------------------------------------------


def generate_readme_split(results, provider, output_dir):

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    onnx_ver = onnx.__version__
    ort_ver = ort.__version__

    try:
        cpu_name = cpuinfo.get_cpu_info().get("brand_raw", platform.processor())
    except Exception:
        cpu_name = platform.processor() or "Unknown"

    gpu_list = []
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True)
        gpu_list = [g.strip() for g in res.stdout.strip().split("\n") if g.strip()]
    except Exception:
        pass

    cuda_version = "Unknown"
    try:
        res = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
        for line in res.stdout.splitlines():
            if "release" in line:
                cuda_version = line.split("release")[-1].split(",")[0].strip()
                break
    except Exception:
        pass

    cudnn_version = get_cudnn_version()
    trt_version = get_tensorrt_version()

    install_cmd = {
        "CUDAExecutionProvider": "pip install onnxruntime-gpu",
        "OpenVINOExecutionProvider": "pip install onnxruntime-openvino\npip install openvino==2025.1.0",
        "DmlExecutionProvider": "pip install onnxruntime-directml",
        "TensorrtExecutionProvider": "manual build with CUDA 12.5, cuDNN 9.2.1, TensorRT 10.9.0.34",
        "DnnlExecutionProvider": "manual build from source (oneDNN included, no pre-install needed)"
    }
    install_info = install_cmd.get(provider, "Installation method not specified")

    def summarize(subset, tag):
        counts = defaultdict(int)
        for _, _, _, status in subset:
            if status.startswith("SUCCESS (with complexification)"):
                counts["Executable directly (SUCCESS with complexification)"] += 1
            elif status.startswith("SUCCESS") and "FALLBACK" not in status:
                counts["SUCCESS"] += 1
            elif "WITH FALLBACK" in status:
                counts["FALLBACK"] += 1
            elif status.startswith("UNKNOWN"):
                counts["UNKNOWN"] += 1
            elif status.startswith("NOT TESTED"):
                counts["NOT TESTED"] += 1
            elif status.startswith("SKIPPED"):
                counts["SKIPPED"] += 1
            elif status.startswith("FAIL"):
                counts["FAIL"] += 1
            else:
                counts["FAIL"] += 1

        total = sum(counts.values()) or 1
        pie_path = os.path.join(output_dir, f"stats_{provider}_{tag}.png")

        fig, ax = plt.subplots(figsize=(6, 6))
        labels = [f"{k}: {counts[k]}" for k in counts]
        colors_map = {
            "SUCCESS": "#00AA44",
            "Executable directly (SUCCESS with complexification)": "#2299DD",
            "FALLBACK": "#FFAA00",
            "UNKNOWN": "#DEDEDE",
            "NOT TESTED": "#4D7CFE",
            "SKIPPED": "#CCCCCC",
            "FAIL": "#FF0000"
        }
        colors = [colors_map.get(k, "#AAAAAA") for k in counts]

        if total > 0:
            ax.pie(counts.values(), labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
            ax.axis('equal')
        else:
            ax.text(0.5, 0.5, "No nodes tested", ha='center', va='center', fontsize=12, color='gray')
            ax.axis('off')

        plt.savefig(pie_path, bbox_inches="tight")
        plt.close(fig)

        return counts, total, pie_path

    def render_section(title, subset, tag):
        lines = [f"## {title}", "", "| ONNX Node | Status |", "|:---------:|:------:|"]
        for op_name, _, _, status in subset:
            if status.startswith("SUCCESS (with complexification)"):
                label = "SUCCESS"
            elif status.startswith("SUCCESS WITH FALLBACK"):
                label = "FALLBACK"
            elif status.startswith("SUCCESS"):
                label = "SUCCESS"
            elif status.startswith("FAIL"):
                label = "FAIL"
            elif status.startswith("NOT TESTED"):
                label = "NOT TESTED"
            elif status.startswith("SKIPPED"):
                label = "SKIPPED"
            else:
                label = "UNKNOWN"

            color = {
                "SUCCESS": "00AA44",
                "FALLBACK": "FFAA00",
                "FAIL": "FF0000",
                "NOT TESTED": "7777CC",
                "SKIPPED": "999999",
                "UNKNOWN": "AAAAAA"
            }.get(label, "000000")

            badge = f"![{label}](https://img.shields.io/badge/{label.replace(' ', '%20')}-{color}?style=flat&logoColor=white)"
            link = f"https://onnx.ai/onnx/operators/onnx__{op_name}.html"
            lines.append(f"| [`{op_name}`]({link}) | {badge} |")

        counts, total, pie_path = summarize(subset, tag)
        lines += [
            "",
            f"### Statistics",
            f"- **Total nodes tested:** {total}",
            f"- **Executable directly (SUCCESS):** {counts['SUCCESS']} ({round((counts['SUCCESS']/total)*100, 1)}%)",
            f"- **Executable directly (SUCCESS with complexification):** {counts['Executable directly (SUCCESS with complexification)']} ({round((counts['Executable directly (SUCCESS with complexification)']/total)*100, 1)}%)",
            f"- **Executable via FALLBACK:** {counts['FALLBACK']} ({round((counts['FALLBACK']/total)*100, 1)}%)",
            f"- **UNKNOWN (no Node event):** {counts['UNKNOWN']} ({round((counts['UNKNOWN']/total)*100, 1)}%)",
            f"- **NOT TESTED:** {counts['NOT TESTED']} ({round((counts['NOT TESTED']/total)*100, 1)}%)",
            f"- **SKIPPED:** {counts['SKIPPED']} ({round((counts['SKIPPED']/total)*100, 1)}%)",
            f"- **FAIL:** {counts['FAIL']} ({round((counts['FAIL']/total)*100, 1)}%)",
            "",
            f"![Pie Chart](./{os.path.basename(pie_path)})",
            ""
        ]
        return lines

    basic_results = [r for r in results if not r[0].startswith("com.microsoft.")]
    ms_results = [r for r in results if r[0].startswith("com.microsoft.")]

    os.makedirs(output_dir, exist_ok=True)
    lines = []
    lines.append(f"# ONNXRuntime Test Results — Provider: `{provider}`\n")
    lines.append(f"**Test Date:** {now}\n")
    if provider == "OpenVINOExecutionProvider":
        lines.append("## Test Methodology  \n"
                 "Each ONNX operator is tested using a minimal ONNX model containing only that specific node.  \n"
                 "In cases where OpenVINO falls back to CPU for simple models, we re-test the node with a slightly complexified "
                 "model. This model adds a chain of `Mul` or `And` operations (based on input type) that preserve the behavior "
                 "but help OpenVINO recognize and execute the subgraph. This allows better detection of real OpenVINO support.\n")
    else:
        lines.append("## Test Methodology  \n"
                 "Each ONNX operator is tested individually using a minimal ONNX model containing only that specific node. "
                 "This ensures a focused and isolated evaluation of operator support for the selected Execution Provider.\n")
    lines.append("### Test Configuration\n")
    lines.append("- **ONNX Opset version:** 22")
    lines.append("- **ONNX IR version:** 10")
    lines.append("- **Data types:** Only one type is tested per node. This is usually `float32`, unless the node does not support it — in which case a compatible type is selected.\n")

    lines.append("## Environment and Installation Details\n")
    lines.append(f"- **ONNX version:** {onnx_ver}")
    lines.append(f"- **ONNXRuntime version:** {ort_ver}")
    lines.append(f"- **Target provider:** {provider}")
    lines.append(f"- **Installation command:**\n```bash\n{install_info}\n```")
    lines.append("### Hardware and Software Versions\n")
    lines.append(f"- **CPU:** {cpu_name}")
    if provider in ["CUDAExecutionProvider", "TensorrtExecutionProvider"] and gpu_list:
        lines.append(f"- **GPU(s):** {', '.join(gpu_list)}")
        lines.append(f"- **CUDA version:** {cuda_version}")
        lines.append(f"- **cuDNN version:** {cudnn_version}")
        if provider == "TensorrtExecutionProvider":
            lines.append(f"- **TensorRT version:** {trt_version}")
    lines.append("")

    lines += render_section("Basic ONNX Nodes", basic_results, "basic")
    lines += render_section("Microsoft Custom Nodes", ms_results, "ms")

    skipped = [op for op, _, _, st in results if "SKIPPED" in st]
    not_tested = [op for op, _, _, st in results if st.startswith("NOT TESTED")]

    if not_tested:
        lines.append("## Nodes not tested\n")
        lines.append("These nodes couldn't be tested due to lack of valid minimal ONNX model.\n")
        lines.append(", ".join(f"`{n}`" for n in sorted(not_tested)))
        lines.append("")
    if skipped:
        lines.append("### Nodes skipped due to runtime errors\n")
        lines.append(", ".join(f"`{n}`" for n in sorted(skipped)))
        lines.append("")

    lines.append("## README Generation\n")
    lines.append("This file was generated automatically by `report.py`.\n")
    lines.append("- Generated ONNX models: `models/<provider>/`")
    lines.append("- Profiling JSON files: `profiling/<provider>/`")
    lines.append("- Scripts: `main.py`, `report.py`, `utils.py`, `ops/*`")
    lines.append("_End of README_")

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Split README written to: {readme_path}")



def generate_full_readme():
    import urllib.parse

    source_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(source_dir)
    output_path = os.path.join(project_root, "README.md")

    regex = {
    "SUCCESS": re.compile(r"\*\*Executable directly \(SUCCESS\):\*\*\s*(\d+)", re.IGNORECASE),
    "SUCCESS (with complexification)": re.compile(r"\*\*Executable directly \(SUCCESS with complexification\):\*\*\s*(\d+)", re.IGNORECASE),
    "FALLBACK": re.compile(r"\*\*Executable via FALLBACK:\*\*\s*(\d+)", re.IGNORECASE),
    "FAIL": re.compile(r"\*\*FAIL:\*\*\s*(\d+)", re.IGNORECASE),
    "NOT TESTED": re.compile(r"\*\*NOT TESTED:\*\*\s*(\d+)", re.IGNORECASE),
    "SKIPPED": re.compile(r"\*\*SKIPPED:\*\*\s*(\d+)", re.IGNORECASE),
    }




    def parse_section(readme, title=None):
        if title:
            # Tolère emojis ou espaces en plus dans le titre
            pattern = re.compile(rf"## .*{re.escape(title)}.*\n", re.IGNORECASE)
            matches = list(pattern.finditer(readme))
            if not matches:
                return None
            start = matches[0].end()
            next_section = re.search(r"^## .+", readme[start:], re.MULTILINE)
            end = start + next_section.start() if next_section else len(readme)
            section = readme[start:end]
        else:
            section = readme
        results = {k: 0 for k in regex}
        for k, pattern in regex.items():
            matches = pattern.findall(section)
            for m in matches:
                results[k] += int(m)
        return [results[k] for k in regex]



    ep_folders = sorted([
        d for d in os.listdir(project_root)
        if os.path.isdir(os.path.join(project_root, d)) and os.path.isfile(os.path.join(project_root, d, "README.md"))
    ])

    rows_basic = []
    rows_ms = []

    for ep in ep_folders:
        readme_path = os.path.join(project_root, ep, "README.md")
        with open(readme_path, encoding="utf-8") as f:
            content = f.read()

        # Basic ONNX Nodes
        basic = parse_section(content, "Basic ONNX Nodes")
        if not basic:
            basic = parse_section(content)  # fallback global

        # Microsoft Custom Nodes
        ms = parse_section(content, "Microsoft Custom Nodes")
        if not ms:
            ms = [0] * 5

        def format_row(data):
            total = sum(data) or 1
            supported = data[0] + data[1] + data[2]  # ajoute "SUCCESS with complexification"
            return [
                f"{data[0] + data[1]} ({round(100*(data[0] + data[1])/total)}%)",  # SUCCESS total
                f"{data[2]} ({round(100*data[2]/total)}%)",  # FALLBACK
                f"{supported} ({round(100*supported/total)}%)",
                f"{data[3]} ({round(100*data[3]/total)}%)",  # FAIL
                f"{data[4]} ({round(100*data[4]/total)}%)",  # NOT TESTED
                f"{data[5]} ({round(100*data[5]/total)}%)",  # SKIPPED
            ]

        rows_basic.append([ep] + format_row(basic))
        rows_ms.append([ep] + format_row(ms))

    #print("rows_basic : ", rows_basic)
    cpu_name = platform.processor() or "Unknown"
    try:
        import cpuinfo
        cpu_name = cpuinfo.get_cpu_info().get("brand_raw", cpu_name)
    except Exception:
        pass

    gpu_list = []
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
        gpu_list = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
    except Exception:
        pass

    cuda_version = "Unknown"
    try:
        res = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if "release" in line:
                cuda_version = line.split("release")[-1].split(",")[0].strip()
                break
    except Exception:
        pass

    cudnn_version = get_cudnn_version()
    trt_version = get_tensorrt_version()

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"""<div style="font-family:Arial, sans-serif; line-height:1.6; max-width:900px; margin:auto; padding:20px;">

<p align="center">
  <img src="https://github.com/microsoft/onnxruntime/raw/main/docs/images/ONNX_Runtime_logo_dark.png" alt="ONNX Runtime Logo" width="320"/>
</p>

<h1>Welcome to the ONNX Runtime – Execution Provider Coverage Tester</h1>

<p>
  This open source initiative, led by <strong><a href="https://graiphic.io/" target="_blank">Graiphic</a></strong>, provides 
  a detailed, real-world coverage map of ONNX operator support for each <strong>Execution Provider (EP)</strong> in 
  <strong><a href="https://github.com/microsoft/onnxruntime" target="_blank">ONNX Runtime</a></strong>.
</p>

<p>
  It is part of our broader effort to democratize AI deployment through 
  <a href="https://graiphic.io/download/" target="_blank"><strong>SOTA</strong></a> — 
  an ONNX-native orchestration framework designed for engineers, researchers, and industrial use cases.
</p>

<h2>🎯 Project Objectives</h2>
<ul>
  <li>Systematically test and report ONNX operator coverage per Execution Provider.</li>
  <li>Deliver up-to-date insights to guide industrial and academic ONNX Runtime adoption.</li>
  <li>Help developers, maintainers, and hardware vendors prioritize missing or broken operator support.</li>
</ul>

<h2>🧪 What’s Tested</h2>
<ul>
  <li>Each ONNX operator is tested in isolation across all supported EPs.</li>
  <li>Results include status per operator: <code>SUCCESS</code>, <code>FALLBACK</code>, <code>FAIL</code>, <code>NOT TESTED</code>, <code>SKIPPED</code>, <code>UNKNOWN</code>.</li>
  <li>Each EP includes a complete dataset with test logs, node-level breakdowns, and global stats.</li>
</ul>

<h2>📐 How’s Tested</h2>
<p>
  Each operator is tested using a <strong>minimal ONNX model</strong> containing only the node under test and its required inputs.
  These models are constructed dynamically for each operator, and executed with the target <strong>Execution Provider (EP)</strong>.
</p>
<p>
  In most cases, this provides a direct and unambiguous signal of EP support: if a node runs successfully in isolation, it is considered
  <strong>natively supported</strong>.
</p>
<p>
  However, for some EPs such as <strong>OpenVINO</strong> and <strong>TensorRT</strong>, fallback to CPU may occur even if the node is technically supported.
  This can be due to backend heuristics requiring a minimal graph complexity to activate EP-specific execution paths.
  In such cases, we attempt a <strong>model complexification step</strong> by embedding the node in a richer context (e.g., with dummy operations).
  If the node then executes on the EP, its status is upgraded to <code>SUCCESS (with complexification)</code>.
</p>

<h3>⚙️ Test Configuration</h3>
<ul>
  <li><strong>ONNX Opset version:</strong> 22</li>
  <li><strong>ONNX IR version:</strong> 10</li>
  <li><strong>Data types:</strong> Only a single data type is tested per node. In general this is <code>float32</code>,
      unless the node does not support it—in which case an available type is selected.</li>
</ul>

<h2>📦 Currently Supported Execution Providers</h2>
<ul>
""")
        for row in rows_basic:
            ep = row[0]
            ep_url = urllib.parse.quote(ep)
            ep_link = f'<a href="./{ep_url}/" target="_blank">{ep}</a>'
            out.write(f"<li>{ep_link}</li>\n")
        out.write('<li><em>Coming soon:</em> AMD – Vitis AI and ROCm</li>\n</ul>\n')

        out.write(f"""
<h2>📊 Summary of ONNX Execution Provider Results</h2>
<p>This summary reflects the latest test run on the following configuration:</p>
<ul>
  <li><strong>CPU:</strong> {cpu_name}</li>
  <li><strong>GPU:</strong> {', '.join(gpu_list) if gpu_list else 'No NVIDIA GPU detected'}</li>
  <li><strong>CUDA:</strong> {cuda_version} | <strong>cuDNN:</strong> {cudnn_version} | <strong>TensorRT:</strong> {trt_version}</li>
  <li><strong>ONNX:</strong> {onnx.__version__} | <strong>ONNXRuntime:</strong> {ort.__version__}</li>
  <li><strong>OS:</strong> {platform.system()} {platform.release()}</li>
</ul>
""")

        def write_table(title, rows):
            out.write(f"<h3>{title}</h3>\n")
            out.write("""<table border="1" cellpadding="6" cellspacing="0">
  <thead>
    <tr>
      <th>Execution Provider</th>
      <th>SUCCESS</th>
      <th>FALLBACK</th>
      <th>SUPPORTED</th>
      <th>FAIL</th>
      <th>NOT TESTED</th>
      <th>SKIPPED</th>
    </tr>
  </thead>
  <tbody>\n""")
            for row in rows:
                ep = row[0]
                ep_url = urllib.parse.quote(ep)
                ep_link = f'<a href="./{ep_url}/" target="_blank">{ep}</a>'
                out.write(f"<tr><td>{ep_link}</td><td>{row[1]}</td><td>{row[2]}</td><td>{row[3]}</td><td>{row[4]}</td><td>{row[5]}</td><td>{row[6]}</td></tr>\n")
            out.write("</tbody></table>\n")

        write_table("ONNX Core Operators", rows_basic)
        out.write("""
<p>
  The ONNX Core Operators table reflects the support status of standard <strong>ONNX specification operators</strong>. 
  These are the official operators defined by the ONNX community, and represent the majority of common model operations.
</p>
""")

        write_table("Microsoft Custom Operators", rows_ms)
        out.write("""
<p>
  The Microsoft Custom Operators table lists <strong>proprietary or domain-specific extensions</strong> provided by ONNX Runtime, 
  typically used in models exported from tools like Olive, Azure ML, or other Microsoft pipelines.
</p>
""")
        out.write("""
<h4>Legend:</h4>
<ul>
  <li><strong>SUCCESS</strong>: Node executed natively by the Execution Provider.</li>
  <li><strong>FALLBACK</strong>: Node executed by a fallback mechanism (typically CPU).</li>
  <li><strong>SUPPORTED</strong>: Sum of SUCCESS and FALLBACK, indicates total operability.</li>
  <li><strong>FAIL</strong>: Node failed execution even with fallback enabled.</li>
  <li><strong>NOT TESTED</strong>: Node was not tested for this provider (unsupported type or config).</li>
  <li><strong>SKIPPED</strong>: Node was deliberately skipped due to known incompatibility or user choice.</li>
</ul>
""")


        out.write("""
<h2>🤝 Maintainer</h2>
<p>
  This project is maintained by <strong><a href="https://graiphic.io/" target="_blank">Graiphic</a></strong> 
  as part of the <a href="https://graiphic.io/download/" target="_blank"><strong>SOTA</strong></a> initiative.
</p>
<p>
  We welcome collaboration, community feedback, and open contribution to make ONNX Runtime stronger and more widely adopted.
</p>

<p style="margin-top:20px;">
  📬 <strong>Contact:</strong> <a href="mailto:contact@graiphic.io">contact@graiphic.io</a><br>
  🌐 <strong>Website:</strong> <a href="https://graiphic.io/" target="_blank">graiphic.io</a><br>
  🧠 <strong>Learn more about SOTA:</strong> <a href="https://graiphic.io/download/" target="_blank">graiphic.io/download</a>
</p>

</div>""")

    print(f"✅ Full README with fallback handling written to: {output_path}")
