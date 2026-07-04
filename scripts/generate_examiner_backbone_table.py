#!/usr/bin/env python3
"""Generate LaTeX tables comparing examiner backbone models for VG and SVG datasets."""

import pandas as pd
import os

BASE = "/raid/william/project/context-eval-mllm/work_dirs"

# Model name mapping (CSV key -> display name)
MODEL_DISPLAY = {
    "gemini_gemini-2.5-flash-image": "Gemini-2.5-Flash",
    "openai_gpt-5.4-mini": "GPT-5.4-mini",
    "zhipu_glm-5v-turbo": "GLM-5V-Turbo",
    # SVG variants
    "gemini_gemini-2.5-flash-image_svg": "Gemini-2.5-Flash",
    "openai_gpt-5.4-mini_svg": "GPT-5.4-mini",
    "zhipu_glm-5v-turbo_svg": "GLM-5V-Turbo",
}
REF_MODEL_VG = "openai_gpt-4o"
REF_MODEL_SVG = "openai_gpt-4o_svg"

BACKBONE_ORDER_VG = [
    "gemini_gemini-2.5-flash-image",
    "openai_gpt-5.4-mini",
    "zhipu_glm-5v-turbo",
]
BACKBONE_ORDER_SVG = [
    "gemini_gemini-2.5-flash-image_svg",
    "openai_gpt-5.4-mini_svg",
    "zhipu_glm-5v-turbo_svg",
]


def load_csv(path):
    df = pd.read_csv(path).set_index("model")
    return df


def delta_str(val, ref):
    d = (val - ref) * 100
    sign = "+" if d >= 0 else ""
    color = "green!60!black" if d >= 0 else "red!70!black"
    arrow = r"$\triangle$" if d >= 0 else r"$\triangledown$"
    return rf"\textcolor{{{color}}}{{{arrow}}} {sign}{d:.1f}\%"


def build_rows(df_cedi, df_v19, backbones, ref_key, suffix=""):
    rows = []
    ref_cedi_chair = df_cedi.loc[ref_key, "overall_CHAIRi_v2"]
    ref_cedi_cov = df_cedi.loc[ref_key, "overall_Coverage_avg"]
    ref_v19_chair = df_v19.loc[ref_key, "overall_CHAIRi_v2"]
    ref_v19_cov = df_v19.loc[ref_key, "overall_Coverage_avg"]

    # CEDI group rows
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{3}{l}{\textit{CEDI (GPT-4o backbone = ref)}} \\")
    for m in backbones:
        if m not in df_cedi.index:
            continue
        name = MODEL_DISPLAY.get(m, m)
        chair = delta_str(df_cedi.loc[m, "overall_CHAIRi_v2"], ref_cedi_chair)
        cov = delta_str(df_cedi.loc[m, "overall_Coverage_avg"], ref_cedi_cov)
        rows.append(rf"{name} & {chair} & {cov} \\")

    # CEDI_v19 group rows
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{3}{l}{\textit{CEDI\_v19 (GPT-4o backbone = ref)}} \\")
    for m in backbones:
        if m not in df_v19.index:
            continue
        name = MODEL_DISPLAY.get(m, m)
        chair = delta_str(df_v19.loc[m, "overall_CHAIRi_v2"], ref_v19_chair)
        cov = delta_str(df_v19.loc[m, "overall_Coverage_avg"], ref_v19_cov)
        rows.append(rf"{name} & {chair} & {cov} \\")

    return rows


def make_table(rows, caption, label):
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\tiny",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Model & CHAIRi fix ($\uparrow$) & Cov avg ($\uparrow$) \\",
        r"\midrule",
    ]
    lines += rows
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    # VG
    vg_cedi = load_csv(f"{BASE}/vg/ablation_baseline2_api_completed/chair_results_summary.csv")
    vg_v19 = load_csv(f"{BASE}/vg/v19_api_completed/chair_results_summary.csv")
    vg_rows = build_rows(vg_cedi, vg_v19, BACKBONE_ORDER_VG, REF_MODEL_VG)
    table_vg = make_table(
        vg_rows,
        "Examiner backbone comparison on VG: delta vs.\ CEDI (GPT-4o).",
        "tab:exam-backbone-vg",
    )

    # SVG
    svg_cedi = load_csv(f"{BASE}/svg/ablation_baseline2_api/chair_results_summary.csv")
    svg_v19 = load_csv(f"{BASE}/svg/v19_api/chair_results_summary.csv")
    # Drop _cache rows
    svg_cedi = svg_cedi[~svg_cedi.index.str.endswith("_cache")]
    svg_v19 = svg_v19[~svg_v19.index.str.endswith("_cache")]
    svg_rows = build_rows(svg_cedi, svg_v19, BACKBONE_ORDER_SVG, REF_MODEL_SVG)
    table_svg = make_table(
        svg_rows,
        "Examiner backbone comparison on SVG: delta vs.\ CEDI (GPT-4o).",
        "tab:exam-backbone-svg",
    )

    print("% ===== VG TABLE =====")
    print(table_vg)
    print()
    print("% ===== SVG TABLE =====")
    print(table_svg)


if __name__ == "__main__":
    main()
