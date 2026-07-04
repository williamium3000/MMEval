#!/usr/bin/env python3
"""
Generate two LaTeX main results tables (VG and SVG) comparing caption / baseline / CEDI
examiner schemes across VLM models.

Metrics populated from data:
  - CHAIRi fix  : overall_CHAIRi_v2   × 100  (from chair_results_summary.csv)
  - Cov avg     : overall_Coverage_avg × 100  (from chair_results_summary.csv)
  - mmhal       : overall_hallucination_rate × 100 (from mmhal_results_summary*.csv; VG only)

Metrics left as --- (not yet computed):
  - GED, DELCON, SoftSp, HaELM, Faith
"""

import os
import glob
import pandas as pd

BASE = "/raid/william/project/context-eval-mllm/work_dirs"

# ── directory config ─────────────────────────────────────────────────────────

VG_DIRS = {
    "caption": f"{BASE}/vg/caption",
    "baseline": f"{BASE}/vg/ablation_baseline2_completed",
    "CEDI": f"{BASE}/vg/final_run_v18_gpt4o_completed",
    "CEDI_v20": f"{BASE}/vg/final_run_v20_gpt4o_small",
}
VG_API_DIRS = {
    "caption": f"{BASE}/vg/caption_api_completed",
    "baseline": f"{BASE}/vg/ablation_baseline2_api_completed",
    "CEDI_v19": f"{BASE}/vg/v19_api_completed",
}
SVG_DIRS = {
    "caption": f"{BASE}/svg/caption",
    "baseline": f"{BASE}/svg/ablation_baseline2",
    # No CEDI run available yet for SVG
}
SVG_API_DIRS = {
    "caption": f"{BASE}/svg/caption_api",
    "baseline": f"{BASE}/svg/ablation_baseline2_api",
    "CEDI_v19": f"{BASE}/svg/v19_api",
}

# ── model display names (dir name → (line1, line2)) ─────────────────────────

MODEL_DISPLAY = {
    "llava-1.5-7b-hf":        ("\\textbf{LLaVA-1.5}", "\\textbf{7B}"),
    "InternVL2-8B":            ("\\textbf{InternVL2}", "\\textbf{8B}"),
    "InternVL2_5-8B":          ("\\textbf{InternVL2.5}", "\\textbf{8B}"),
    "InternVL3-8B-Instruct":   ("\\textbf{InternVL3}", "\\textbf{8B Instruct}"),
    "Qwen2.5-VL-7B-Instruct":  ("\\textbf{Qwen2.5-VL-7B}", "\\textbf{Instruct}"),
    "gemma-3-12b-it":          ("\\textbf{gemma-3-it}", "\\textbf{12B}"),
    "opera-llava-1.5":         ("\\textbf{opera-llava}", "\\textbf{1.5}"),
}

# API model names in VG dirs → display names
# SVG uses a "_svg" suffix; strip it for lookup
API_MODEL_DISPLAY = {
    "gemini_gemini-2.5-flash-image": ("\\textbf{Gemini-2.5}", "\\textbf{Flash}"),
    "openai_gpt-4o":                 ("\\textbf{GPT-4o}", ""),
    "openai_gpt-5.4-mini":           ("\\textbf{GPT-5.4}", "\\textbf{mini}"),
    "zhipu_glm-5v-turbo":            ("\\textbf{GLM-5V}", "\\textbf{Turbo}"),
}
# SVG key = VG key + "_svg"  (strip suffix when looking up display name)
API_SVG_SUFFIX = "_svg"

# preferred order
MODEL_ORDER = list(MODEL_DISPLAY.keys())
API_MODEL_ORDER = list(API_MODEL_DISPLAY.keys())

# ── helpers ──────────────────────────────────────────────────────────────────

def load_chair(dirpath: str) -> pd.DataFrame:
    """Load chair_results_summary.csv from dirpath, return indexed by model."""
    p = os.path.join(dirpath, "chair_results_summary.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p).set_index("model")
    return df


def load_mmhal(dirpath: str) -> pd.DataFrame:
    """
    Load mmhal results from dirpath.
    Tries:
      1. mmhal_results_summary.csv  (VG caption aggregated)
      2. mmhal_results_summary_*.csv (CEDI aggregated)
    Returns DataFrame indexed by model with column overall_hallucination_rate.
    """
    # 1. aggregated in root
    p = os.path.join(dirpath, "mmhal_results_summary.csv")
    if os.path.exists(p):
        return pd.read_csv(p).set_index("model")

    # 2. single aggregated file with dir-name suffix
    pattern = os.path.join(dirpath, "mmhal_results_summary_*.csv")
    files = glob.glob(pattern)
    if files:
        frames = [pd.read_csv(f) for f in files]
        combined = pd.concat(frames).set_index("model")
        return combined

    return pd.DataFrame()


def pct(val) -> str:
    """Format a 0-1 float as a percentage string like '31.1\\%'."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "---"
    return f"{val * 100:.1f}\\%"


def _numeric(s: str):
    """Return float from a pct string, or None if ---."""
    if s == "---":
        return None
    return float(s.replace("\\%", ""))


def bold_best(values: list[str], higher_is_better: bool = True) -> list[str]:
    """
    Bold the best value in a list of pct strings (skipping ---).
    Applies \\textbf{} to best cell.
    """
    nums = [(_numeric(v), i) for i, v in enumerate(values)]
    valid = [(n, i) for n, i in nums if n is not None]
    if not valid:
        return values
    best_val = max(n for n, _ in valid) if higher_is_better else min(n for n, _ in valid)
    result = list(values)
    for n, i in valid:
        if n == best_val:
            result[i] = f"\\textbf{{{values[i]}}}"
    return result


def get_model_key(chair_df: pd.DataFrame, model_name: str):
    """Look up model_name or its _cache/_svg/_svg_cache variant in chair_df index."""
    if model_name in chair_df.index:
        return model_name
    for suffix in ("_cache", "_svg", "_svg_cache"):
        candidate = model_name + suffix
        if candidate in chair_df.index:
            return candidate
    return None


def get_metrics(chair_df: pd.DataFrame, mmhal_df: pd.DataFrame, model_name: str):
    """
    Returns (chair_fix, cov_avg, mmhal) as pct strings for a given model.
    """
    key = get_model_key(chair_df, model_name)
    if key is None or chair_df.empty:
        return "---", "---", "---"

    chair_fix = pct(chair_df.loc[key, "overall_CHAIRi_v2"])
    cov_avg = pct(chair_df.loc[key, "overall_Coverage_avg"])

    mmhal_val = "---"
    if not mmhal_df.empty:
        mkey = get_model_key(mmhal_df, model_name)
        if mkey is not None:
            mmhal_val = pct(mmhal_df.loc[mkey, "overall_hallucination_rate"])

    return chair_fix, cov_avg, mmhal_val


# ── table builder ────────────────────────────────────────────────────────────

EXAMINER_LABELS = {
    "caption": "caption",
    "baseline": "baseline",
    "CEDI": "CEDI",
    "CEDI_v19": "CEDI\\_v19",
    "CEDI_v20": "CEDI\\_v20",
}

def _build_model_rows(
    model_name: str,
    display: tuple[str, str],
    schemes: list[str],
    chairs: dict,
    mmhals: dict,
) -> list[str]:
    """Return LaTeX lines for one model group (all examiner rows)."""
    display_cell = f"\\parbox{{2.5cm}}{{\\centering{display[0]}\\\\ {display[1]}}}"

    rows: dict[str, tuple] = {}
    for scheme in schemes:
        rows[scheme] = get_metrics(chairs[scheme], mmhals[scheme], model_name)

    # bold best per metric column (all three are higher=better)
    for col_idx in range(3):
        vals = [rows[s][col_idx] for s in schemes]
        bolded = bold_best(vals, higher_is_better=True)
        for s, bv in zip(schemes, bolded):
            rows[s] = tuple(bv if i == col_idx else rows[s][i] for i in range(3))

    n = len(schemes)
    lines = []
    s0 = schemes[0]
    r0 = rows[s0]
    lines.append(
        f"\\multirow{{{n}}}{{*}}{{{display_cell}}} "
        f"& {EXAMINER_LABELS[s0]} "
        f"& {r0[0]} & {r0[1]} & {r0[2]} "
        f"& --- & --- & --- & --- & --- \\\\"
    )
    for s in schemes[1:]:
        r = rows[s]
        is_cedi = s in ("CEDI", "CEDI_v19", "CEDI_v20")
        prefix = (
            "\\rowcolor[HTML]{E6F2FF}\n"
            " \\multicolumn{1}{l}{\\cellcolor[HTML]{E6F2FF}}"
            if is_cedi else " "
        )
        lines.append(
            f"{prefix} & {EXAMINER_LABELS[s]} "
            f"& {r[0]} & {r[1]} & {r[2]} "
            f"& --- & --- & --- & --- & --- \\\\"
        )
    return lines


def build_table_body(
    dirs: dict[str, str],
    api_dirs: dict[str, str] | None = None,
) -> list[str]:
    """Build \\midrule-separated row groups for local + API models."""
    chairs = {scheme: load_chair(d) for scheme, d in dirs.items()}
    mmhals = {scheme: load_mmhal(d) for scheme, d in dirs.items()}
    schemes = list(dirs.keys())

    api_chairs, api_mmhals, api_schemes = {}, {}, []
    if api_dirs:
        api_chairs = {scheme: load_chair(d) for scheme, d in api_dirs.items()}
        api_mmhals = {scheme: load_mmhal(d) for scheme, d in api_dirs.items()}
        api_schemes = list(api_dirs.keys())

    lines = []

    # ── local (open) models ──────────────────────────────────────────────────
    for model_name in MODEL_ORDER:
        model_lines = _build_model_rows(
            model_name, MODEL_DISPLAY[model_name], schemes, chairs, mmhals
        )
        lines.extend(model_lines)
        lines.append("\\midrule")

    # ── API models ────────────────────────────────────────────────────────────
    if api_dirs:
        for api_key in API_MODEL_ORDER:
            model_lines = _build_model_rows(
                api_key, API_MODEL_DISPLAY[api_key],
                api_schemes, api_chairs, api_mmhals,
            )
            lines.extend(model_lines)
            lines.append("\\midrule")

    # drop trailing \midrule
    if lines and lines[-1] == "\\midrule":
        lines[-1] = ""
    return lines


def wrap_table(body_lines: list[str], caption: str, label: str) -> str:
    header = r"""\begin{table*}[t]
\centering
\tiny
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\begin{tabular}{@{}l|l|cccccccc@{}}
\toprule
\textbf{Model} & \textbf{Examiner} & CHAIRi fix ($\uparrow$) & Cov avg ($\uparrow$) & mmhal ($\uparrow$) & GED ($\uparrow$) & DELCON ($\uparrow$) & SoftSp ($\downarrow$) & HaELM ($\uparrow$) & Faith ($\downarrow$) \\
\midrule"""
    footer = r"""\end{tabular}
\end{table*}"""

    body = "\n".join(body_lines)
    return f"{header}\n{body}\n\\bottomrule\n{footer}"


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # VG table
    vg_body = build_table_body(VG_DIRS, api_dirs=VG_API_DIRS)
    vg_table = wrap_table(
        vg_body,
        "Main results on VG by examiner scheme (100 caption, 100 dyna sg, 100 baseline).",
        "tab:main-exam-vg",
    )

    # SVG table
    svg_body = build_table_body(SVG_DIRS, api_dirs=SVG_API_DIRS)
    svg_table = wrap_table(
        svg_body,
        "Main results on SVG by examiner scheme.",
        "tab:main-exam-svg",
    )

    print("% ===== VG TABLE =====")
    print(vg_table)
    print()
    print("% ===== SVG TABLE =====")
    print(svg_table)


if __name__ == "__main__":
    main()
