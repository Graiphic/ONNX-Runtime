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
    import cpuinfo  # to get a more readable CPU name
except ImportError:
    cpuinfo = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#from utils import map_execution_provider


# --------------------------------------------------
# Excel Report Generation: Detailed (non-aggregated)
# --------------------------------------------------

def generate_report(results, provider, profiling_dir, models_dir):
    """
    Generates an Excel file report_{provider}.xlsx from the `results` list.
    Distinguishes these categories:
      - SUCCESS
      - SUCCESS (via decomposition)
      - SUCCESS WITH FALLBACK
      - SUCCESS WITH FALLBACK (via decomposition)
      - UNKNOWN (no Node event)
      - FAIL
    """

    # 1) Create workbook and main sheet
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Report"

    # Column headers
    headers = ["op", "provider", "used_provider", "status"]
    ws.append(headers)

    # 2) Define color mapping for each status (English labels)
    color_map = {
        "SUCCESS": "00AA44",                                  # green
        "SUCCESS (via decomposition)": "007700",              # dark green
        "SUCCESS WITH FALLBACK": "FFAA00",                    # orange
        "SUCCESS WITH FALLBACK (via decomposition)": "FF7700",# dark orange
        "UNKNOWN (no Node event)": "DEDEDE",                  # light gray
        "FAIL": "FF0000",                                     # red
    }

    def get_fill_for_status(status):
        """
        Returns a PatternFill for the given status.
        """
        if status == "SUCCESS":
            return PatternFill(fill_type="solid", start_color=color_map["SUCCESS"], end_color=color_map["SUCCESS"])
        elif status == "SUCCESS (via decomposition)":
            return PatternFill(fill_type="solid", start_color=color_map["SUCCESS (via decomposition)"], end_color=color_map["SUCCESS (via decomposition)"])
        elif status == "SUCCESS WITH FALLBACK":
            return PatternFill(fill_type="solid", start_color=color_map["SUCCESS WITH FALLBACK"], end_color=color_map["SUCCESS WITH FALLBACK"])
        elif status == "SUCCESS WITH FALLBACK (via decomposition)":
            return PatternFill(fill_type="solid", start_color=color_map["SUCCESS WITH FALLBACK (via decomposition)"], end_color=color_map["SUCCESS WITH FALLBACK (via decomposition)"])
        elif status == "UNKNOWN (no Node event)":
            return PatternFill(fill_type="solid", start_color=color_map["UNKNOWN (no Node event)"], end_color=color_map["UNKNOWN (no Node event)"])
        elif status.startswith("FAIL"):
            return PatternFill(fill_type="solid", start_color=color_map["FAIL"], end_color=color_map["FAIL"])
        else:
            # Default to red for any unexpected status
            return PatternFill(fill_type="solid", start_color=color_map["FAIL"], end_color=color_map["FAIL"])

    # 3) Write results into the "Report" sheet
    for row in results:
        op_name, provider_used, used_provider, status = row
        ws.append([op_name, provider_used, used_provider or "", status])
        fill = get_fill_for_status(status)
        cell_status = ws.cell(row=ws.max_row, column=4)
        cell_status.fill = fill

    # 4) Auto-resize columns A–D based on content (max width = 80)
    for col_idx in range(1, 5):  # columns A (1) to D (4)
        max_length = 0
        col_letter = get_column_letter(col_idx)
        for cell in ws[col_letter]:
            if cell.value:
                length = len(str(cell.value))
                if length > max_length:
                    max_length = length
        ws.column_dimensions[col_letter].width = min(max_length + 2, 80)

    # 5) Add thin borders around each cell in A–D
    thin_side = Side(style="thin", color="000000")
    thin_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=4):
        for cell in row:
            cell.border = thin_border

    # 6) Prepare data for the pie chart (detailed categories)
    counter = Counter()
    for _, _, _, status in results:
        if status in color_map and status != "FAIL":
            counter[status] += 1
        elif status.startswith("FAIL"):
            counter["FAIL"] += 1
        else:
            counter["FAIL"] += 1

    # Create a sheet "Data_PieChart" with Category, Count, Percentage
    ws_data = wb.create_sheet(title="Data_PieChart")
    ws_data.append(["Category", "Count", "Percentage"])

    total_count = sum(counter.values()) or 1
    for key in [
        "SUCCESS",
        "SUCCESS (via decomposition)",
        "SUCCESS WITH FALLBACK",
        "SUCCESS WITH FALLBACK (via decomposition)",
        "UNKNOWN (no Node event)",
        "FAIL",
    ]:
        count = counter.get(key, 0)
        percent = round((count / total_count) * 100, 1)
        ws_data.append([key, count, percent])

    # 6.1) Add timestamp below the data table
    timestamp_row = ws_data.max_row + 1
    ws_data.cell(row=timestamp_row, column=1, value="Report generated on")
    ws_data.cell(row=timestamp_row, column=2, value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Auto-resize columns in the data sheet (A–C)
    for col_idx in range(1, 4):
        max_length = 0
        col_letter = get_column_letter(col_idx)
        for cell in ws_data[col_letter]:
            if cell.value is not None:
                length = len(str(cell.value))
                if length > max_length:
                    max_length = length
        ws_data.column_dimensions[col_letter].width = max_length + 2

    # 7) Create the pie chart with percentages
    chart = PieChart()
    chart.title = "Status Distribution"
    chart.title.overlay = False
    chart.width = 20
    chart.height = 12

    # Define categories (A2:A7) and values (B2:B7)
    cats = Reference(ws_data, min_col=1, min_row=2, max_row=7)
    vals = Reference(ws_data, min_col=2, min_row=2, max_row=7)
    chart.add_data(vals, titles_from_data=False)
    chart.set_categories(cats)
    chart.legend.position = 'r'

    # Color each slice according to color_map
    series = chart.series[0]
    colors_ordered = [
        color_map["SUCCESS"],
        color_map["SUCCESS (via decomposition)"],
        color_map["SUCCESS WITH FALLBACK"],
        color_map["SUCCESS WITH FALLBACK (via decomposition)"],
        color_map["UNKNOWN (no Node event)"],
        color_map["FAIL"],
    ]
    for idx, col_hex in enumerate(colors_ordered):
        dp = DataPoint(
            idx=idx,
            spPr=GraphicalProperties(solidFill=ColorChoice(srgbClr=col_hex))
        )
        series.dPt.append(dp)

    # Display percentages on each slice
    chart.dataLabels = DataLabelList()
    chart.dataLabels.showPercent = True

    # Adjust plot area to push the pie chart to the left
    chart.plot_area.layout = Layout(
        manualLayout=ManualLayout(
            x=0.00,
            y=0.20,
            w=0.50,
            h=0.60
        )
    )

    # Insert chart into the "Report" sheet at cell F2
    ws.add_chart(chart, "F2")

    # 8) Save the workbook
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
      - SUCCESS (includes "SUCCESS (via decomposition)")
      - SUCCESS WITH FALLBACK (includes "SUCCESS WITH FALLBACK (via decomposition)")
      - UNKNOWN (no Node event)
      - FAIL
    """

    # 1) Create workbook and main sheet
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Report"

    # Column headers
    headers = ["op", "provider", "used_provider", "status"]
    ws.append(headers)

    # 2) Define color mapping for each aggregated status
    agg_color_map = {
        "SUCCESS": "00AA44",                  # green
        "SUCCESS WITH FALLBACK": "FFAA00",    # orange
        "UNKNOWN (no Node event)": "DEDEDE",  # light gray
        "FAIL": "FF0000",                     # red
    }

    def get_fill_for_agg_status(status):
        """
        Returns a PatternFill for the aggregated status.
        """
        if status.startswith("SUCCESS") and "FALLBACK" not in status:
            # Covers "SUCCESS" and "SUCCESS (via decomposition)"
            return PatternFill(fill_type="solid", start_color=agg_color_map["SUCCESS"], end_color=agg_color_map["SUCCESS"])
        elif "WITH FALLBACK" in status:
            # Covers "SUCCESS WITH FALLBACK" and "SUCCESS WITH FALLBACK (via decomposition)"
            return PatternFill(fill_type="solid", start_color=agg_color_map["SUCCESS WITH FALLBACK"], end_color=agg_color_map["SUCCESS WITH FALLBACK"])
        elif status == "UNKNOWN (no Node event)":
            return PatternFill(fill_type="solid", start_color=agg_color_map["UNKNOWN (no Node event)"], end_color=agg_color_map["UNKNOWN (no Node event)"])
        elif status.startswith("FAIL"):
            return PatternFill(fill_type="solid", start_color=agg_color_map["FAIL"], end_color=agg_color_map["FAIL"])
        else:
            return PatternFill(fill_type="solid", start_color=agg_color_map["FAIL"], end_color=agg_color_map["FAIL"])

    # 3) Write results into the "Report" sheet with aggregated coloring
    for row in results:
        op_name, provider_used, used_provider, status = row
        ws.append([op_name, provider_used, used_provider or "", status])
        fill = get_fill_for_agg_status(status)
        cell_status = ws.cell(row=ws.max_row, column=4)
        cell_status.fill = fill

    # 4) Auto-resize columns A–D
    for col_idx in range(1, 5):
        max_length = 0
        col_letter = get_column_letter(col_idx)
        for cell in ws[col_letter]:
            if cell.value:
                length = len(str(cell.value))
                if length > max_length:
                    max_length = length
        ws.column_dimensions[col_letter].width = min(max_length + 2, 80)

    # 5) Add thin borders around each cell in A–D
    thin_side = Side(style="thin", color="000000")
    thin_border = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=4):
        for cell in row:
            cell.border = thin_border

    # 6) Compute aggregated counts for the pie chart
    counter_agg = Counter()
    for _, _, _, status in results:
        if status.startswith("SUCCESS") and "FALLBACK" not in status:
            counter_agg["SUCCESS"] += 1
        elif "WITH FALLBACK" in status:
            counter_agg["SUCCESS WITH FALLBACK"] += 1
        elif status == "UNKNOWN (no Node event)":
            counter_agg["UNKNOWN (no Node event)"] += 1
        elif status.startswith("FAIL"):
            counter_agg["FAIL"] += 1
        else:
            counter_agg["FAIL"] += 1

    # Create a sheet "Data_PieChart" with Category, Count, Percentage
    ws_data = wb.create_sheet(title="Data_PieChart")
    ws_data.append(["Category", "Count", "Percentage"])

    total_agg = sum(counter_agg.values()) or 1
    for key in [
        "SUCCESS",
        "SUCCESS WITH FALLBACK",
        "UNKNOWN (no Node event)",
        "FAIL",
    ]:
        count = counter_agg.get(key, 0)
        percent = round((count / total_agg) * 100, 1)
        ws_data.append([key, count, percent])

    # 6.1) Add timestamp below the data table
    timestamp_row = ws_data.max_row + 1
    ws_data.cell(row=timestamp_row, column=1, value="Report generated on")
    ws_data.cell(row=timestamp_row, column=2, value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Auto-resize columns A–C
    for col_idx in range(1, 4):
        max_length = 0
        col_letter = get_column_letter(col_idx)
        for cell in ws_data[col_letter]:
            if cell.value is not None:
                length = len(str(cell.value))
                if length > max_length:
                    max_length = length
        ws_data.column_dimensions[col_letter].width = max_length + 2

    # 7) Create the pie chart for aggregated categories
    chart = PieChart()
    chart.title = "Status Distribution (aggregated)"
    chart.title.overlay = False
    chart.width = 20
    chart.height = 12

    cats = Reference(ws_data, min_col=1, min_row=2, max_row=5)
    vals = Reference(ws_data, min_col=2, min_row=2, max_row=5)
    chart.add_data(vals, titles_from_data=False)
    chart.set_categories(cats)
    chart.legend.position = 'r'

    series = chart.series[0]
    colors_agg = [
        agg_color_map["SUCCESS"],
        agg_color_map["SUCCESS WITH FALLBACK"],
        agg_color_map["UNKNOWN (no Node event)"],
        agg_color_map["FAIL"],
    ]
    for idx, col_hex in enumerate(colors_agg):
        dp = DataPoint(
            idx=idx,
            spPr=GraphicalProperties(solidFill=ColorChoice(srgbClr=col_hex))
        )
        series.dPt.append(dp)

    chart.dataLabels = DataLabelList()
    chart.dataLabels.showPercent = True

    chart.plot_area.layout = Layout(
        manualLayout=ManualLayout(
            x=0.00,
            y=0.20,
            w=0.50,
            h=0.60
        )
    )

    ws.add_chart(chart, "F2")

    # 8) Save the workbook
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
      - Test date + statistics (direct / fallback / not executable / fail).
      - A pie chart PNG showing the distribution.
      - A detailed list of all nodes (name | status), each status as a colored badge.
      - Information on how each Execution Provider was installed (including DNNL).
      - Versions of ONNX / ONNXRuntime / CUDA / cuDNN / TensorRT.
      - Hardware info: CPU name and GPU(s).
      - Relative links to scripts and folders.
    """
    # --- 1) Compute statistics ---
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(results) if results else 1

    count_direct = 0
    count_fallback = 0
    count_unknown = 0
    count_fail = 0

    for _, _, _, status in results:
        if ("SUCCESS" in status) and ("FALLBACK" not in status):
            count_direct += 1
        elif "SUCCESS WITH FALLBACK" in status:
            count_fallback += 1
        elif status == "UNKNOWN (no Node event)":
            count_unknown += 1
        elif status.startswith("FAIL"):
            count_fail += 1
        else:
            count_fail += 1

    pct_direct = round((count_direct / total) * 100, 1)
    pct_fallback = round((count_fallback / total) * 100, 1)
    pct_unknown = round((count_unknown / total) * 100, 1)
    pct_fail = round((count_fail / total) * 100, 1)

    # --- 2) ONNX / ONNXRuntime versions ---
    onnx_ver = onnx.__version__
    ort_ver = ort.__version__

    # --- 3) CPU information ---
    if cpuinfo:
        try:
            cpu_name = cpuinfo.get_cpu_info().get("brand_raw", platform.processor())
        except Exception:
            cpu_name = platform.processor() or "Unknown"
    else:
        cpu_name = platform.processor() or "Unknown"

    # --- 4) GPU information via nvidia-smi ---
    gpu_list = []
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        raw_gpu = res.stdout.strip().split("\n")
        gpu_list = [g.strip() for g in raw_gpu if g.strip()]
    except Exception:
        gpu_list = []

    # --- 5) CUDA version via nvcc ---
    cuda_version = "Unknown"
    try:
        res = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=True
        )
        for line in res.stdout.splitlines():
            if "release" in line:
                part = line.split("release")[-1].split(",")[0].strip()
                cuda_version = part
                break
    except Exception:
        cuda_version = "Unknown"

    # --- 6) cuDNN version by reading the header file in CUDA ---
    cudnn_version = get_cudnn_version()

    # --- 7) TensorRT version ---
    trt_version = get_tensorrt_version()

    # --- 8) Determine installation command by provider ---
    installation_cmd = {
        "CUDAExecutionProvider":      "pip install onnxruntime-gpu",
        "OpenVINOExecutionProvider":  "pip install onnxruntime-openvino\npip install openvino==2025.1.0",
        "DmlExecutionProvider":       "pip install onnxruntime-directml",
        "TensorrtExecutionProvider":  "manual build with CUDA 12.5, cuDNN 9.2.1, TensorRT 10.9.0.34",
        "DnnlExecutionProvider":      "manual build from source (oneDNN included, no pre-install needed)"
    }
    install_info = installation_cmd.get(provider, "Installation method not specified")

    # --- 9) Generate pie chart PNG ---
    labels = [
        f"Direct (SUCCESS): {count_direct}",
        f"Fallback: {count_fallback}",
        f"Unknown: {count_unknown}",
        f"Fail: {count_fail}"
    ]
    sizes = [count_direct, count_fallback, count_unknown, count_fail]
    colors = ["#00AA44", "#FFAA00", "#DEDEDE", "#FF0000"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors
    )
    ax.axis('equal')

    os.makedirs(output_dir, exist_ok=True)
    pie_path = os.path.join(output_dir, f"stats_{provider}.png")
    plt.savefig(pie_path, bbox_inches="tight")
    plt.close(fig)

    # --- 10) Build Markdown content (English) ---
    # Map each status to its hex color
    status_color_map = {
        "SUCCESS": "#00AA44",
        "SUCCESS (via decomposition)": "#007700",
        "SUCCESS WITH FALLBACK": "#FFAA00",
        "SUCCESS WITH FALLBACK (via decomposition)": "#FF7700",
        "UNKNOWN (no Node event)": "#DEDEDE",
        "FAIL": "#FF0000",
    }

    lines = []
    lines.append(f"# ONNXRuntime Test Results — Provider: `{provider}`")
    lines.append("")
    lines.append(f"**Test Date:** {now}")
    lines.append("")
    lines.append("## Global Statistics")
    lines.append("")
    lines.append(f"- **Total nodes tested:** {total}")
    lines.append(f"- **Executable directly (SUCCESS):** {count_direct} ({pct_direct}%)")
    lines.append(f"- **Executable via FALLBACK:** {count_fallback} ({pct_fallback}%)")
    lines.append(f"- **UNKNOWN (no Node event):** {count_unknown} ({pct_unknown}%)")
    lines.append(f"- **FAIL:** {count_fail} ({pct_fail}%)")
    lines.append("")
    lines.append("### Statistics Pie Chart")
    lines.append("")
    rel_pie = os.path.basename(pie_path)
    lines.append(f"![Node Status Distribution](./{rel_pie})")
    lines.append("")
    lines.append("## Node Details")
    lines.append("")
    lines.append("| ONNX Node | Status |")
    lines.append("|:---------:|:------:|")

    for op_name, _, _, status in results:
        # If status starts with "FAIL", display only "FAIL"
        if status.startswith("FAIL"):
            display_label = "FAIL"
        else:
            display_label = status

        # Get the hex color for this display label (default to black if not found)
        hex_color = status_color_map.get(display_label, "#000000")

        # URL‐encode spaces in the label (for shields.io badge)
        label_encoded = display_label.replace(" ", "%20")
        badge_url = (
            f"https://img.shields.io/badge/{label_encoded}-{hex_color.lstrip('#')}?style=flat&logoColor=white"
        )
        badge_md = f"![{display_label}]({badge_url})"

        # --- UPDATED PART: link the operator name to its ONNX doc ---
        # Build a Markdown link like [`Add`](https://onnx.ai/onnx/operators/onnx__Add.html)
        # NOTE: we keep the code font around the op_name by wrapping it in backticks inside the link.
        op_url = f"https://onnx.ai/onnx/operators/onnx__{op_name}.html"
        node_link = f"[`{op_name}`]({op_url})"

        # Final table row
        lines.append(f"| {node_link} | {badge_md} |")
    lines.append("")

    lines.append("## Environment and Installation Details")
    lines.append("")
    lines.append(f"- **ONNX version:** {onnx_ver}")
    lines.append(f"- **ONNXRuntime version:** {ort_ver}")
    lines.append(f"- **Target provider:** {provider}")
    lines.append(f"- **Installation command for this provider:**\n```bash\n{install_info}\n```")
    lines.append("")
    lines.append("### Hardware and Software Versions")
    lines.append("")
    lines.append(f"- **CPU:** {cpu_name}")
    if gpu_list:
        gpu_str = ", ".join(gpu_list)
        lines.append(f"- **GPU(s):** {gpu_str}")
        lines.append(f"- **CUDA version:** {cuda_version}")
        lines.append(f"- **cuDNN version:** {cudnn_version}")
        if provider == "TensorrtExecutionProvider":
            lines.append(f"- **TensorRT version:** {trt_version}")
    else:
        lines.append("- **GPU:** No NVIDIA GPU detected")
        lines.append(f"- **CUDA version:** {cuda_version}")
        lines.append(f"- **cuDNN version:** {cudnn_version}")
        if provider == "TensorrtExecutionProvider":
            lines.append(f"- **TensorRT version:** {trt_version}")
    lines.append("")
    lines.append("### System Information")
    lines.append("")
    os_info = platform.system() + " " + platform.release()
    lines.append(f"- **Operating System (OS):** {os_info}")
    lines.append("")
    lines.append("## README Generation")
    lines.append("")
    lines.append("This file was automatically generated by `report.py` using the `generate_readme` function.")
    lines.append("")
    lines.append("### Related Scripts and Folders")
    lines.append("")
    lines.append("- Generated ONNX models: `models/<provider>/`")
    lines.append("- Profiling JSON files: `profiling/<provider>/`")
    lines.append("- Test scripts: `main.py`, `utils.py`, `report.py`, `ops/*`")
    lines.append("")
    lines.append("_End of README_")
    lines.append("")

    # --- 11) Write the README to disk ---
    readme_path = os.path.join(output_dir, f"README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated README: {readme_path}")
    print(f"Pie chart PNG saved as: {pie_path}")


def generate_root_summary():
    """
    Parcourt tous les dossiers situés à la racine du projet (au-dessus de Source/).
    Pour chaque dossier qui contient un README.md, extrait SUCCESS, FALLBACK et FAIL,
    calcule les pourcentages, puis écrit un README.md unique à la racine du projet,
    avec en plus une section décrivant la méthodologie de test et un résumé du
    matériel et logiciels utilisés.
    """

    # 1) Déterminer le chemin du dossier racine (celui qui contient Source/)
    source_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(source_dir)

    # 2) Regex pour extraire les trois métriques du README de chaque EP
    SUCCESS_REGEX  = re.compile(r"Executable\s+directly.*?:\s*\*\*\s*(\d+)\b", re.IGNORECASE)
    FALLBACK_REGEX = re.compile(r"Executable\s+via\s+FALLBACK.*?:\s*\*\*\s*(\d+)\b", re.IGNORECASE)
    FAIL_REGEX     = re.compile(r"FAIL.*?[:\s]*\*\*\s*(\d+)\b", re.IGNORECASE)

    # 3) Collecter une liste de tuples : (nom_dossier, success, fallback, fail)
    table_rows = []
    for entry in os.listdir(project_root):
        folder_path = os.path.join(project_root, entry)
        if not os.path.isdir(folder_path):
            continue

        readme_path = os.path.join(folder_path, "README.md")
        if not os.path.isfile(readme_path):
            continue

        # Par défaut : initialiser à zéro
        success_count  = 0
        fallback_count = 0
        fail_count     = 0

        contenu = ""
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                contenu = f.read()
        except Exception:
            contenu = ""

        # Extraction des trois compteurs
        m = SUCCESS_REGEX.search(contenu)
        if m:
            success_count = int(m.group(1))

        m = FALLBACK_REGEX.search(contenu)
        if m:
            fallback_count = int(m.group(1))

        m = FAIL_REGEX.search(contenu)
        if m:
            fail_count = int(m.group(1))

        # Calcul du total et des pourcentages
        total = success_count + fallback_count + fail_count
        if total == 0:
            pct_succ = pct_fb = pct_fail = 0
        else:
            pct_succ = round((success_count / total) * 100)
            pct_fb   = round((fallback_count / total) * 100)
            pct_fail = round((fail_count / total) * 100)

        # Conserver : nom_dossier et chaînes "count (pct%)"
        table_rows.append((
            entry,
            f"{success_count} ({pct_succ}%)",
            f"{fallback_count} ({pct_fb}%)",
            f"{fail_count} ({pct_fail}%)"
        ))

    # 4) Si aucun dossier EP détecté, on stoppe
    if not table_rows:
        print("Aucun dossier EP trouvé à la racine pour générer un résumé.")
        return

    # 5) Préparer le chemin du README global
    output_path = os.path.join(project_root, "README.md")
    with open(output_path, "w", encoding="utf-8") as out:

        # 5.a) Titre et paragraphe introductif
        out.write("# Summary of ONNX Execution Provider Results\n\n")
        out.write(
            "This document gathers all per-provider test results in one place.  "
            "Below is a table summarizing, for each Execution Provider (EP), "
            "how many nodes ran successfully on the provider directly (SUCCESS), "
            "how many fell back (FALLBACK), and how many failed (FAIL), "
            "each with its percentage of the total.  "
            "Refer to each EP’s own README for detailed charts and further statistics.\n\n"
        )

        # 5.b) Tableau Markdown
        out.write("| Execution Provider | SUCCESS | FALLBACK | FAIL |\n")
        out.write("|:------------------:|:-------:|:--------:|:----:|\n")
        for ep_name, succ_str, fb_str, fail_str in table_rows:
            out.write(f"| {ep_name} | {succ_str} | {fb_str} | {fail_str} |\n")
        out.write("\n")

        # 5.c) Section: Testing Methodology
        out.write("## Testing Methodology\n\n")
        out.write(
            "Each ONNX operator (node) is tested by dynamically building a minimal ONNX model containing only that single node.  "
            "The test harness then creates an ONNXRuntime inference session with profiling enabled, "
            "runs the model once, and inspects the generated profiling trace to determine whether the node was executed successfully.  "
            "From the profiling events, we also identify which Execution Provider (EP) actually processed the node—"
            "either directly on the target EP or via a fallback path.  "
            "If the node did not produce any 'Node' event in the profiler output, it is marked as UNKNOWN; "
            "otherwise, it is classified as SUCCESS (direct) or FALLBACK (if ONNXRuntime fell back to another EP).\n\n"
        )

        # 5.d) Section: Hardware and Software
        out.write("## Hardware and Software\n\n")

        # -- CPU Info --
        if cpuinfo:
            try:
                cpu_name = cpuinfo.get_cpu_info().get("brand_raw", platform.processor())
            except Exception:
                cpu_name = platform.processor() or "Unknown"
        else:
            cpu_name = platform.processor() or "Unknown"
        out.write(f"- **CPU:** {cpu_name}\n")

        # -- GPU Info via nvidia-smi --
        gpu_list = []
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, check=True
            )
            raw_gpu = res.stdout.strip().split("\n")
            gpu_list = [g.strip() for g in raw_gpu if g.strip()]
        except Exception:
            gpu_list = []

        if gpu_list:
            out.write(f"- **GPU(s):** {', '.join(gpu_list)}\n")
        else:
            out.write("- **GPU(s):** No NVIDIA GPU detected\n")

        # -- CUDA Version via nvcc --
        cuda_version = "Unknown"
        try:
            res = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True, text=True, check=True
            )
            for line in res.stdout.splitlines():
                if "release" in line:
                    cuda_version = line.split("release")[-1].split(",")[0].strip()
                    break
        except Exception:
            cuda_version = "Unknown"
        out.write(f"- **CUDA version:** {cuda_version}\n")

        # -- cuDNN Version via lecture du header --
        def parse_cudnn_header(header_path):
            try:
                with open(header_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                return None
            maj = re.search(r"#define\s+CUDNN_MAJOR\s+(\d+)", content)
            mino = re.search(r"#define\s+CUDNN_MINOR\s+(\d+)", content)
            patch = re.search(r"#define\s+CUDNN_PATCHLEVEL\s+(\d+)", content)
            if maj and mino and patch:
                return f"{maj.group(1)}.{mino.group(1)}.{patch.group(1)}"
            return None

        def get_cudnn_version():
            cuda_env = os.environ.get("CUDA_PATH")
            candidates = []
            if cuda_env:
                candidates.append(cuda_env)
                if os.path.basename(cuda_env).lower().startswith("v"):
                    candidates.append(os.path.dirname(cuda_env))
            prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
            candidate_root = os.path.join(prog_files, "NVIDIA GPU Computing Toolkit", "CUDA")
            candidates.append(candidate_root)
            for base in candidates:
                if not os.path.isdir(base):
                    continue
                include_dir = os.path.join(base, "include")
                if os.path.isdir(include_dir):
                    for fname in ("cudnn_version.h", "cudnn.h"):
                        path_h = os.path.join(include_dir, fname)
                        if os.path.isfile(path_h):
                            ver = parse_cudnn_header(path_h)
                            if ver:
                                return ver
                for sub in os.listdir(base):
                    sub_inc = os.path.join(base, sub, "include")
                    if not os.path.isdir(sub_inc):
                        continue
                    for fname in ("cudnn_version.h", "cudnn.h"):
                        path_h = os.path.join(sub_inc, fname)
                        if os.path.isfile(path_h):
                            ver = parse_cudnn_header(path_h)
                            if ver:
                                return ver
            return "Unknown"

        cudnn_version = get_cudnn_version()
        out.write(f"- **cuDNN version:** {cudnn_version}\n")


        trt_version = get_tensorrt_version()
        out.write(f"- **TensorRT version:** {trt_version}\n")

        # -- ONNX / ONNXRuntime Info --
        onnx_ver = onnx.__version__
        ort_ver  = ort.__version__
        out.write(f"- **ONNX version:** {onnx_ver}\n")
        out.write(f"- **ONNXRuntime version:** {ort_ver}\n")

        # -- OS Info --
        os_info = platform.system() + " " + platform.release()
        out.write(f"- **Operating System (OS):** {os_info}\n")

        out.write("\n")  # ligne vide finale

    print(f"→ Root README.md updated at: {output_path}")
