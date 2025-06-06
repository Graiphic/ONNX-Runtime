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

    # Color mapping
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

    # Fill the main table with rows and colors
    for row in results:
        op_name, provider_used, used_provider, status = row
        ws.append([op_name, provider_used, used_provider or "", status])
        ws.cell(row=ws.max_row, column=4).fill = get_fill_for_status(status)

    # Resize columns
    for col_idx in range(1, 5):
        max_len = max(len(str(cell.value)) for cell in ws[get_column_letter(col_idx)] if cell.value)
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 80)

    # Add borders
    border = Border(left=Side(style="thin"), right=Side(style="thin"),
                    top=Side(style="thin"), bottom=Side(style="thin"))
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=4):
        for cell in row:
            cell.border = border

    # Count each category
    counter = Counter()
    for _, _, _, status in results:
        matched = False
        for key in color_map:
            if status == key or status.startswith(key):
                counter[key] += 1
                matched = True
                break
        if not matched:
            counter["FAIL"] += 1

    # Data sheet with summary
    ws_data = wb.create_sheet("Data_PieChart")
    ws_data.append(["Category", "Count", "Percentage"])
    total = sum(counter.values()) or 1

    ordered_keys = [
        "SUCCESS",
        "SUCCESS (via decomposition)",
        "SUCCESS WITH FALLBACK",
        "SUCCESS WITH FALLBACK (via decomposition)",
        "UNKNOWN (no Node event)",
        "NOT TESTED",
        "SKIPPED",
        "FAIL",
    ]
    for key in ordered_keys:
        count = counter.get(key, 0)
        percent = round(100 * count / total, 1)
        ws_data.append([key, count, percent])

    # Timestamp
    ws_data.cell(row=ws_data.max_row + 1, column=1, value="Report generated on")
    ws_data.cell(row=ws_data.max_row, column=2, value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Resize columns A–C
    for col_idx in range(1, 4):
        max_len = max(len(str(cell.value)) for cell in ws_data[get_column_letter(col_idx)] if cell.value)
        ws_data.column_dimensions[get_column_letter(col_idx)].width = max_len + 2

    # Pie chart
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

    # Save file
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

    # Aggregated color map
    agg_color_map = {
        "SUCCESS": "00AA44",                   # green
        "SUCCESS WITH FALLBACK": "FFAA00",     # orange
        "UNKNOWN (no Node event)": "DEDEDE",   # light gray
        "NOT TESTED": "4D7CFE",                # blue
        "SKIPPED": "CCCCCC",                   # light gray 2
        "FAIL": "FF0000",                      # red
    }

    def get_fill_for_agg_status(status):
        if status.startswith("SUCCESS") and "FALLBACK" not in status:
            return PatternFill("solid", fgColor=agg_color_map["SUCCESS"])
        elif "WITH FALLBACK" in status:
            return PatternFill("solid", fgColor=agg_color_map["SUCCESS WITH FALLBACK"])
        elif status.startswith("UNKNOWN"):
            return PatternFill("solid", fgColor=agg_color_map["UNKNOWN (no Node event)"])
        elif status.startswith("NOT TESTED"):
            return PatternFill("solid", fgColor=agg_color_map["NOT TESTED"])
        elif status.startswith("SKIPPED"):
            return PatternFill("solid", fgColor=agg_color_map["SKIPPED"])
        elif status.startswith("FAIL"):
            return PatternFill("solid", fgColor=agg_color_map["FAIL"])
        return PatternFill("solid", fgColor=agg_color_map["FAIL"])

    # Fill table with colors
    for row in results:
        op_name, provider_used, used_provider, status = row
        ws.append([op_name, provider_used, used_provider or "", status])
        ws.cell(row=ws.max_row, column=4).fill = get_fill_for_agg_status(status)

    # Auto-resize columns
    for col_idx in range(1, 5):
        max_len = max(len(str(cell.value)) for cell in ws[get_column_letter(col_idx)] if cell.value)
        ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 80)

    # Borders
    border = Border(left=Side(style="thin"), right=Side(style="thin"),
                    top=Side(style="thin"), bottom=Side(style="thin"))
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=4):
        for cell in row:
            cell.border = border

    # Status count aggregation
    counter_agg = Counter()
    for _, _, _, status in results:
        if status.startswith("SUCCESS") and "FALLBACK" not in status:
            counter_agg["SUCCESS"] += 1
        elif "WITH FALLBACK" in status:
            counter_agg["SUCCESS WITH FALLBACK"] += 1
        elif status.startswith("UNKNOWN"):
            counter_agg["UNKNOWN (no Node event)"] += 1
        elif status.startswith("NOT TESTED"):
            counter_agg["NOT TESTED"] += 1
        elif status.startswith("SKIPPED"):
            counter_agg["SKIPPED"] += 1
        elif status.startswith("FAIL"):
            counter_agg["FAIL"] += 1
        else:
            counter_agg["FAIL"] += 1

    # Write data summary
    ws_data = wb.create_sheet("Data_PieChart")
    ws_data.append(["Category", "Count", "Percentage"])

    ordered_keys = [
        "SUCCESS",
        "SUCCESS WITH FALLBACK",
        "UNKNOWN (no Node event)",
        "NOT TESTED",
        "SKIPPED",
        "FAIL",
    ]
    total_agg = sum(counter_agg.values()) or 1
    for key in ordered_keys:
        count = counter_agg.get(key, 0)
        percent = round(100 * count / total_agg, 1)
        ws_data.append([key, count, percent])

    # Timestamp
    ts_row = ws_data.max_row + 1
    ws_data.cell(row=ts_row, column=1, value="Report generated on")
    ws_data.cell(row=ts_row, column=2, value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Resize columns
    for col_idx in range(1, 4):
        max_len = max(len(str(cell.value)) for cell in ws_data[get_column_letter(col_idx)] if cell.value)
        ws_data.column_dimensions[get_column_letter(col_idx)].width = max_len + 2

    # Pie chart
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

    # Save
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

def generate_readme(results, provider, output_dir):
    """
    Generates a README_<provider>.md in English containing:
      1) Environment and installation details
      2) Node-level table
      3) Summary stats + pie chart
      4) Skipped and not-tested sections
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(results) if results else 1

    count_direct = count_fallback = count_unknown = count_fail = count_not_tested = count_skipped = 0

    for _, _, _, status in results:
        if status.startswith("SUCCESS") and "FALLBACK" not in status:
            count_direct += 1
        elif "WITH FALLBACK" in status:
            count_fallback += 1
        elif status.startswith("UNKNOWN"):
            count_unknown += 1
        elif status.startswith("NOT TESTED"):
            count_not_tested += 1
        elif status.startswith("SKIPPED"):
            count_skipped += 1
        elif status.startswith("FAIL"):
            count_fail += 1
        else:
            count_fail += 1

    # Calculate percentages
    def pct(x): return round((x / total) * 100, 1)
    pct_direct = pct(count_direct)
    pct_fallback = pct(count_fallback)
    pct_unknown = pct(count_unknown)
    pct_not_tested = pct(count_not_tested)
    pct_skipped = pct(count_skipped)
    pct_fail = pct(count_fail)

    # Version info
    onnx_ver = onnx.__version__
    ort_ver = ort.__version__

    # Hardware info
    if cpuinfo:
        try:
            cpu_name = cpuinfo.get_cpu_info().get("brand_raw", platform.processor())
        except Exception:
            cpu_name = platform.processor() or "Unknown"
    else:
        cpu_name = platform.processor() or "Unknown"

    gpu_list = []
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
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
        "CUDAExecutionProvider":      "pip install onnxruntime-gpu",
        "OpenVINOExecutionProvider":  "pip install onnxruntime-openvino\npip install openvino==2025.1.0",
        "DmlExecutionProvider":       "pip install onnxruntime-directml",
        "TensorrtExecutionProvider":  "manual build with CUDA 12.5, cuDNN 9.2.1, TensorRT 10.9.0.34",
        "DnnlExecutionProvider":      "manual build from source (oneDNN included, no pre-install needed)"
    }
    install_info = install_cmd.get(provider, "Installation method not specified")

    # Pie chart
    labels = [
        f"Direct (SUCCESS): {count_direct}",
        f"Fallback: {count_fallback}",
        f"Unknown: {count_unknown}",
        f"Not Tested: {count_not_tested}",
        f"Skipped: {count_skipped}",
        f"Fail: {count_fail}"
    ]
    sizes = [count_direct, count_fallback, count_unknown, count_not_tested, count_skipped, count_fail]
    colors = ["#00AA44", "#FFAA00", "#DEDEDE", "#4D7CFE", "#CCCCCC", "#FF0000"]

    fig, ax = plt.subplots(figsize=(6, 6))
    if sum(sizes) > 0:
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        ax.axis('equal')
    else:
        ax.text(0.5, 0.5, "No nodes tested", ha='center', va='center', fontsize=12, color='gray')
        ax.axis('off')

    os.makedirs(output_dir, exist_ok=True)
    pie_path = os.path.join(output_dir, f"stats_{provider}.png")
    plt.savefig(pie_path, bbox_inches="tight")
    plt.close(fig)

    # Markdown generation
    status_color_map = {
        "SUCCESS": "#00AA44",
        "SUCCESS (via decomposition)": "#007700",
        "SUCCESS WITH FALLBACK": "#FFAA00",
        "SUCCESS WITH FALLBACK (via decomposition)": "#FF7700",
        "UNKNOWN (no Node event)": "#DEDEDE",
        "NOT TESTED": "#4D7CFE",
        "SKIPPED": "#CCCCCC",
        "FAIL": "#FF0000",
    }

    lines = []
    lines.append(f"# ONNXRuntime Test Results — Provider: `{provider}`\n")
    lines.append(f"**Test Date:** {now}\n")
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

    lines.append("## Node Details")
    lines.append("")
    lines.append("| ONNX Node | Status |")
    lines.append("|:---------:|:------:|")
    
    for op_name, _, _, status in results:
        # Affichage synthétique du statut
        if status.startswith("SUCCESS WITH FALLBACK"):
            display_label = "FALLBACK"
        elif status.startswith("SUCCESS"):
            display_label = "SUCCESS"
        elif status.startswith("FAIL"):
            display_label = "FAIL"
        elif status.startswith("NOT TESTED"):
            display_label = "NOT TESTED"
        elif status.startswith("SKIPPED"):
            display_label = "SKIPPED"
        else:
            display_label = "UNKNOWN"
    
        # Couleur badge
        hex_color = {
            "SUCCESS": "00AA44",
            "FALLBACK": "FFAA00",
            "FAIL": "FF0000",
            "NOT TESTED": "7777CC",
            "SKIPPED": "999999",
            "UNKNOWN": "AAAAAA"
        }.get(display_label, "000000")
    
        label_encoded = display_label.replace(" ", "%20")
        badge_url = f"https://img.shields.io/badge/{label_encoded}-{hex_color}?style=flat&logoColor=white"
        badge_md = f"![{display_label}]({badge_url})"
    
        # Lien vers la doc de l'opérateur
        op_url = f"https://onnx.ai/onnx/operators/onnx__{op_name}.html"
        node_link = f"[`{op_name}`]({op_url})"
    
        lines.append(f"| {node_link} | {badge_md} |")
    
    lines.append("")

    lines.append("## Global Statistics\n")
    lines.append(f"- **Total nodes tested:** {total}")
    lines.append(f"- **Executable directly (SUCCESS):** {count_direct} ({pct_direct}%)")
    lines.append(f"- **Executable via FALLBACK:** {count_fallback} ({pct_fallback}%)")
    lines.append(f"- **UNKNOWN (no Node event):** {count_unknown} ({pct_unknown}%)")
    lines.append(f"- **NOT TESTED:** {count_not_tested} ({pct_not_tested}%)")
    lines.append(f"- **SKIPPED:** {count_skipped} ({pct_skipped}%)")
    lines.append(f"- **FAIL:** {count_fail} ({pct_fail}%)")
    lines.append("\n### Statistics Pie Chart\n")
    lines.append(f"![Node Status Distribution](./{os.path.basename(pie_path)})\n")

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

    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated README: {os.path.join(output_dir, 'README.md')}")
    print(f"Pie chart PNG saved as: {pie_path}")


def generate_root_summary():
    """
    Parses all README.md files under the execution provider folders.
    Extracts the number of nodes with status:
    SUCCESS, FALLBACK, FAIL, NOT TESTED, SKIPPED.
    Then generates a root-level README.md with environment info and summary table.
    """
    source_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(source_dir)

    # Regex patterns for extracting stats
    regex_patterns = {
        "SUCCESS": re.compile(r"Executable\s+directly.*?:\s*\*\*\s*(\d+)\b", re.IGNORECASE),
        "FALLBACK": re.compile(r"Executable\s+via\s+FALLBACK.*?:\s*\*\*\s*(\d+)\b", re.IGNORECASE),
        "FAIL": re.compile(r"\*\*FAIL.*?:\s*\*\*\s*(\d+)\b", re.IGNORECASE),
        "NOT TESTED": re.compile(r"\*\*NOT TESTED.*?:\s*\*\*\s*(\d+)\b", re.IGNORECASE),
        "SKIPPED": re.compile(r"\*\*SKIPPED.*?:\s*\*\*\s*(\d+)\b", re.IGNORECASE),
    }

    # Scan each EP folder
    table_rows = []
    for entry in os.listdir(project_root):
        folder_path = os.path.join(project_root, entry)
        readme_path = os.path.join(folder_path, "README.md")
        if not os.path.isdir(folder_path) or not os.path.isfile(readme_path):
            continue

        # Initialize counters
        counts = {k: 0 for k in regex_patterns.keys()}

        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        for key, pattern in regex_patterns.items():
            match = pattern.search(content)
            if match:
                counts[key] = int(match.group(1))

        total = sum(counts.values()) or 1
        row = [entry]
        for key in ["SUCCESS", "FALLBACK", "FAIL", "NOT TESTED", "SKIPPED"]:
            value = counts[key]
            pct = round((value / total) * 100)
            row.append(f"{value} ({pct}%)")

        table_rows.append(row)

    if not table_rows:
        print("No valid README.md files found for summary.")
        return

    # Write global README
    output_path = os.path.join(project_root, "README.md")
    with open(output_path, "w", encoding="utf-8") as out:
        # Title & intro
        out.write("# Summary of ONNX Execution Provider Results\n\n")
        out.write("This document gathers all test results by Execution Provider (EP).\n")
        out.write("Each EP has its own `README.md` with detailed results.\n\n")

        # Hardware and software
        out.write("## Hardware and Software\n\n")

        # CPU
        if cpuinfo:
            try:
                cpu_name = cpuinfo.get_cpu_info().get("brand_raw", platform.processor())
            except Exception:
                cpu_name = platform.processor() or "Unknown"
        else:
            cpu_name = platform.processor() or "Unknown"
        out.write(f"- **CPU:** {cpu_name}\n")

        # GPU
        gpu_list = []
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, check=True
            )
            gpu_list = [g.strip() for g in res.stdout.strip().split("\n") if g.strip()]
        except Exception:
            pass
        if gpu_list:
            out.write(f"- **GPU(s):** {', '.join(gpu_list)}\n")
        else:
            out.write("- **GPU(s):** No NVIDIA GPU detected\n")

        # CUDA
        cuda_version = "Unknown"
        try:
            res = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
            for line in res.stdout.splitlines():
                if "release" in line:
                    cuda_version = line.split("release")[-1].split(",")[0].strip()
                    break
        except Exception:
            pass
        out.write(f"- **CUDA version:** {cuda_version}\n")

        # cuDNN
        cudnn_version = get_cudnn_version()
        out.write(f"- **cuDNN version:** {cudnn_version}\n")

        # TensorRT
        trt_version = get_tensorrt_version()
        out.write(f"- **TensorRT version:** {trt_version}\n")

        # ONNX/ORT versions
        out.write(f"- **ONNX version:** {onnx.__version__}\n")
        out.write(f"- **ONNXRuntime version:** {ort.__version__}\n")

        # OS
        out.write(f"- **Operating System (OS):** {platform.system()} {platform.release()}\n\n")

        # Summary table
        out.write("## Summary Table\n\n")
        out.write("| Execution Provider | SUCCESS | FALLBACK | FAIL | NOT TESTED | SKIPPED |\n")
        out.write("|:------------------:|:-------:|:--------:|:----:|:-----------:|:--------:|\n")
        for row in table_rows:
            out.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} |\n")
        out.write("\n")

    print(f"→ Root README.md updated at: {output_path}")
