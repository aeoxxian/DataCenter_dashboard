import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
import re
from pathlib import Path
import numpy as np

# Set page config
st.set_page_config(layout="wide", page_title="AI Power Experiment Dashboard", page_icon="⚡")

# Custom CSS
st.markdown("""
<style>
    .exp-header { font-size: 1.2rem; font-weight: 600; color: #1e293b; }
    .exp-desc { color: #64748b; font-size: 0.9rem; margin-bottom: 1rem; }
    .stDataFrame { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# Constants
HYP_OPT_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = HYP_OPT_ROOT / "outputs"
GPU_EXP_DIR = OUTPUTS_ROOT / "gpu_power_experiment"
LLM_EXP_DIR = OUTPUTS_ROOT / "llm_power_experiment"

# ── Experiment Metadata (Updated with Real Folder Names) ────────────────
GPU_EXPERIMENTS = {
    "00_baseline_reference": {"title": "기준 실험 (Baseline)", "desc": "전력 제어 없이 표준 조건으로 실행한 기준 실험입니다.", "baseline_key": "nocap"},
    "00_smoke_test": {"title": "파이프라인 검증 (Smoke Test)", "desc": "측정 파이프라인이 정상 작동하는지 확인하는 빠른 테스트입니다.", "baseline_key": None},
    "01_batch_sweep_resnet50": {"title": "배치 크기 변화 실험", "desc": "배치 크기(16, 64, 128)에 따른 GPU 전력 소비 변화를 관찰합니다.", "baseline_key": "bs64"},
    "02_operator_control": {"title": "연산 제어 실험 (Conv vs Linear)", "desc": "Conv2d와 Linear 레이어의 전력 소비 특성을 비교합니다.", "baseline_key": "conv"},
    "03_control_cap_sweep": {"title": "전력 제한(Power Capping) 실험", "desc": "Power Capping(300W~150W)이 학습 속도와 총 에너지에 미치는 영향을 분석합니다.", "baseline_key": "nocap"},
    "04_control_clock_sweep": {"title": "클럭 고정(Clock Locking) 실험", "desc": "GPU 클럭 주파수 고정이 전력 안정성과 성능에 미치는 영향을 분석합니다.", "baseline_key": "nocap"},
    "05_pattern_fixed_var_burst": {"title": "부하 패턴(Burst) 실험", "desc": "일정한 부하(Fixed)와 간헐적 부하(Burst)의 전력 패턴 차이를 비교합니다.", "baseline_key": "fixed"},
    "06_model_scaling_image": {"title": "모델 크기별 실험 (Vision)", "desc": "ResNet18, ResNet50, ConvNeXt Tiny 등 모델 크기에 따른 전력 특성입니다.", "baseline_key": "resnet50"},
    "07_dataset_cifar10_cifar100_imagenet": {"title": "데이터셋 복잡도 실험", "desc": "CIFAR-10, CIFAR-100 등 데이터셋 복잡도에 따른 전력 차이입니다.", "baseline_key": "cifar10"},
    "08_optimizer_sgd_adam_rmsprop": {"title": "옵티마이저 비교 실험", "desc": "SGD, Adam, RMSprop 옵티마이저별 전력 소비 특성을 비교합니다.", "baseline_key": "sgd"},
    "09_precision_fp32_fp16_bf16": {"title": "정밀도(Precision) 실험", "desc": "FP32, FP16, BF16 정밀도에 따른 전력 절감 효과를 분석합니다.", "baseline_key": "fp32"},
    "10_dataloader_workers_pin_memory": {"title": "데이터로더 최적화 실험", "desc": "Num workers와 Pin memory 설정이 GPU 유휴 시간(Idle)에 미치는 영향입니다.", "baseline_key": "w4_pin"},
    "11_control_clock_sweep": {"title": "클럭 스윕 (Clock Sweep)", "desc": "클럭 주파수를 세밀하게 조절하여 최적의 동작 점을 찾습니다.", "baseline_key": "nocap"}
}

LLM_EXPERIMENTS = {
    "01_model_scaling_initial": {"title": "LLM 모델 크기별 실험 (Initial)", "desc": "GPT2, Qwen 등 LLM 모델 크기에 따른 초기 전력 분석입니다.", "baseline_key": "gpt2"},
    "02_decode_length_32_128": {"title": "디코딩 길이 변화 실험 (Short)", "desc": "출력 토큰 길이(32 vs 128)에 따른 인퍼런스 전력 소비를 비교합니다.", "baseline_key": "maxtok32"},
    "03_dataset_alpaca_mmlu_longbench": {"title": "LLM 데이터셋 비교", "desc": "Alpaca(Inst), MMLU(QA), LongBench(Doc) 데이터셋별 특성입니다.", "baseline_key": "alpaca"},
    "04_decode_length_32_256_512": {"title": "긴 시퀀스 디코딩 실험 (Long)", "desc": "최대 512 토큰까지 디코딩 길이를 늘려가며 전력 패턴을 관찰합니다.", "baseline_key": "maxtok32"},
    "05_inference_pattern_fixed_var_burst": {"title": "LLM 인퍼런스 패턴 실험", "desc": "지속적 요청(Fixed)과 간헐적 요청(Burst)의 전력 효율을 비교합니다.", "baseline_key": "fixed"},
    "06_power_cap_200_240_280_320_360": {"title": "LLM 전력 제한 실험", "desc": "LLM 인퍼런스 시 Power Capping의 효과와 성능 저하를 분석합니다.", "baseline_key": "nocap"},
    "07_clock_lock_1005_1500_2100": {"title": "LLM 클럭 고정 실험", "desc": "GPU 클럭 고정이 LLM 토큰 생성 속도와 전력에 미치는 영향입니다.", "baseline_key": "nocap"},
    "08_ramp_rate": {"title": "Ramp Rate 실험", "desc": "Ramp-up 속도 조절이 전력 피크에 미치는 영향을 분석합니다.", "baseline_key": None},
    "09_control_combo_cap_clock_ramp": {"title": "전력/클럭 복합 제어 실험", "desc": "Cap, Clock, Ramp-up 제어를 조합하여 최적의 설정을 탐색합니다.", "baseline_key": "nocap"},
    "10_precision_bf16_fp16_4bit": {"title": "LLM 정밀도 비교 (FP16/BF16/4bit)", "desc": "FP16과 BF16, 4bit 양자화의 전력 효율 및 메모리 사용량을 비교합니다.", "baseline_key": "fp16"},
    "11_train_infer_split": {"title": "학습 vs 인퍼런스 분리 실험", "desc": "LLM의 학습(Fine-tuning)과 인퍼런스 단계를 분리하여 측정합니다.", "baseline_key": None},
    "12_model_scaling_tokenpowerbench": {"title": "토큰당 전력 벤치마크", "desc": "다양한 모델의 Token당 에너지(J/Token) 효율을 벤치마킹합니다.", "baseline_key": "gpt2"}
}

PHASE_COLORS = {
    "idle": "#94a3b8", "idle_pre": "#cbd5e1", "idle_post": "#64748b", "idle_mid": "#9ca3af",
    "warmup": "#fbbf24", "train": "#ef4444", "train_compute": "#b91c1c",
    "val": "#3b82f6", "validation": "#3b82f6", "test": "#10b981",
    "prefill": "#f59e0b", "decode": "#6366f1", "inference": "#8b5cf6",
    "inference_prefill": "#f59e0b", "inference_decode": "#6366f1", "inference_idle": "#94a3b8",
}
PHASE_LABELS = { k: k.replace("_", " ").title() for k in PHASE_COLORS.keys() } # Simple auto-labeling

def parse_run_info(run_name, config, run_path, context_exp_id=None):
    """
    Context-Aware Parsing: Extracts Key Variables based on Experiment ID.
    Returns: dict with 'Model', 'Condition', 'GroupKey', 'Repeats'
    """
    info = {"Model": "Unknown", "Condition": "Standard", "GroupKey": run_name}
    
    # 1. Base Model Name Extraction
    match = re.search(r'(?:llm|fixed)_(.+?)_', run_name)
    if match:
        raw_model = match.group(1)
        raw_model = raw_model.replace("gpu0", "").replace("ds", "").strip("_")
        info["Model"] = raw_model
    if config and "model_name" in config: info["Model"] = config["model_name"]
    elif config and "model" in config: info["Model"] = config["model"]

    # 2. Key Variable Extraction based on Experiment Context
    conditions = []
    
    # Global Checks (Always relevant if present)
    is_no_cap = "nocap" in run_name or (config and config.get("power_cap") is None)
    
    if context_exp_id:
        # --- VISION ---
        if "01_batch_sweep" in context_exp_id:
            if config and "batch_size" in config: conditions.append(f"BS {config['batch_size']}")
            else: # Fallback to filename
                m = re.search(r'bs(\d+)', run_name)
                if m: conditions.append(f"BS {m.group(1)}")
                
        elif "03_control_cap" in context_exp_id:
            if is_no_cap: conditions.append("No Cap")
            elif config and config.get("power_cap"): conditions.append(f"Cap {config['power_cap']}W")
            else:
                 m = re.search(r'cap(\d+)', run_name)
                 if m: conditions.append(f"Cap {m.group(1)}W")

        elif "04_control_clock" in context_exp_id:
            if config and config.get("clock_lock_gpu"): conditions.append(f"Clock {config['clock_lock_gpu']}MHz")
            else: conditions.append("No Lock")

        elif "06_model_scaling" in context_exp_id:
            # Model name IS the condition basically, but we can be explicit
            pass 

        elif "09_precision" in context_exp_id:
            if config:
                if config.get("use_fp16"): conditions.append("FP16")
                elif config.get("use_bf16"): conditions.append("BF16")
                elif config.get("use_amp"): conditions.append("AMP")
                else: conditions.append("FP32")
                
        # --- LLM ---
        elif "02_decode_length" in context_exp_id or "04_decode_length" in context_exp_id:
            if config and "gen_max_new_tokens" in config:
                conditions.append(f"MaxTok {config['gen_max_new_tokens']}")
                
        elif "05_inference_pattern" in context_exp_id:
            pat = config.get("inference_pattern", "fixed") if config else "fixed"
            if pat == "concurrent_burst": conditions.append("Concurrent Burst")
            elif pat == "variable": conditions.append("Variable")
            elif "burst" in run_name: conditions.append("Burst")
            else: conditions.append("Fixed")
            
        elif "06_power_cap" in context_exp_id:
            if is_no_cap: conditions.append("No Cap")
            elif config and config.get("power_cap"): conditions.append(f"Cap {config['power_cap']}W")
            
        elif "07_clock_lock" in context_exp_id:
            if config and config.get("clock_lock_gpu"): conditions.append(f"Clock {config['clock_lock_gpu']}MHz")
            else: conditions.append("Default Clock")

        elif "10_precision" in context_exp_id:
            if config:
                if config.get("use_4bit"): conditions.append("4bit")
                elif config.get("use_bf16"): conditions.append("BF16")
                elif config.get("use_fp16"): conditions.append("FP16")
                else: conditions.append("FP32")

    # If no specific condition found, fallback to generic parsing
    if not conditions:
        if is_no_cap: conditions.append("No Cap")
        if "fixed" in run_name: conditions.append("Fixed")
    
    # Deduplicate
    unique_conditions = sorted(list(set(conditions)))
    if unique_conditions:
        info["Condition"] = ", ".join(unique_conditions)
    else:
        info["Condition"] = "Standard"

    # Repeats Handling
    parts = run_path.parts
    is_repeat = False
    repeat_idx = ""
    for part in parts:
        if re.match(r'^r\d+$', part):
            is_repeat = True
            repeat_idx = part
            break
            
    info["GroupKey"] = f"{info['Model']} ({info['Condition']})"
    if is_repeat:
        info["Repeats"] = repeat_idx

    return info

@st.cache_data
def load_experiments(base_dir, experiment_meta):
    experiments = {}
    if not base_dir.exists(): return experiments

    for exp_id, meta in experiment_meta.items():
        entry = base_dir / exp_id
        if entry.exists() and entry.is_dir():
            experiments[exp_id] = { "meta": meta, "path": entry, "runs": [] }
            
            # Recursive glob for config.json (handles repeats)
            for run_dir in sorted(entry.rglob("config.json")):
                run_path = run_dir.parent
                
                # Load Config
                try:
                    with open(run_dir) as f: config = json.load(f)
                except: config = {}

                # Load Metrics
                metrics = {}
                metrics_path = run_path / "samples" / "phase_power_summary.csv"
                if not metrics_path.exists(): metrics_path = run_path / "phase_power_summary.csv"
                if metrics_path.exists():
                    try:
                        df = pd.read_csv(metrics_path)
                        if "power_avg_w" in df.columns: metrics = df.set_index("phase").to_dict(orient="index")
                    except: pass
                
                # Load Inference Metrics
                infer_metrics = {}
                infer_path = run_path / "samples" / "inference_results.csv"
                if infer_path.exists():
                    try:
                        df_infer = pd.read_csv(infer_path)
                        if "output_tokens" in df_infer.columns:
                            infer_metrics["total_output_tokens"] = df_infer["output_tokens"].sum()
                    except: pass

                # Context-Aware Parsing
                parsed_info = parse_run_info(run_path.name, config, run_path, context_exp_id=exp_id)

                gpu_samples = run_path / "gpu_samples_lite.csv"
                if not gpu_samples.exists(): gpu_samples = run_path / "samples" / "gpu_samples.csv"
                if not gpu_samples.exists(): gpu_samples = run_path / "gpu_samples.csv"

                experiments[exp_id]["runs"].append({
                    "name": run_path.name,
                    "label": run_path.name, # Raw label
                    "parsed_info": parsed_info, # Rich info
                    "path": run_path,
                    "metrics": metrics,
                    "infer_metrics": infer_metrics,
                    "config": config,
                    "gpu_samples": gpu_samples
                })
    return experiments

@st.cache_data
def load_gpu_samples(csv_path):
    if not csv_path.exists(): return pd.DataFrame()
    return pd.read_csv(csv_path)

def calculate_delta(val, base_val, lower_is_better=True):
    if base_val is None or base_val == 0: return None
    delta = ((val - base_val) / base_val) * 100
    return delta

def create_phase_plot(df, label):
    fig = go.Figure()
    if df.empty or "phase" not in df.columns: return fig
    t_start = df["timestamp_ms"].min()
    df["time"] = (df["timestamp_ms"] - t_start) / 1000.0
    df['phase_group'] = (df['phase'] != df['phase'].shift()).cumsum()
    added_legends = set()
    for _, group in df.groupby('phase_group'):
        phase = group['phase'].iloc[0]
        color = PHASE_COLORS.get(phase, "#000000")
        name = PHASE_LABELS.get(phase, phase)
        show_legend = name not in added_legends
        if show_legend: added_legends.add(name)
        fig.add_trace(go.Scatter(
            x=group["time"], y=group["power_w"], mode='lines', name=name,
            line=dict(color=color, width=2.5), legendgroup=name, showlegend=show_legend, hoverinfo='x+y+name'
        ))
    fig.update_layout(
        title=dict(text=label, font=dict(size=14, color="#1e293b")),
        height=450, margin=dict(l=20, r=20, t=40, b=40),
        xaxis=dict(title="Time (s)", showgrid=True), yaxis=dict(title="Power (W)", showgrid=True),
        plot_bgcolor="white", legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"), hovermode="x unified"
    )
    return fig

def create_comparison_chart(runs, category):
    grouped_data = {}
    for r in runs:
        parsed = r["parsed_info"]
        key = parsed["GroupKey"]
        if key not in grouped_data:
            grouped_data[key] = {"train": [], "infer": [], "model": parsed["Model"], "condition": parsed["Condition"]}
            
        train_pwr = 0
        if "train" in r["metrics"]: train_pwr = r["metrics"]["train"]["power_avg_w"]
        elif "train_compute" in r["metrics"]: train_pwr = r["metrics"]["train_compute"]["power_avg_w"]
        
        infer_pwr = 0
        if "inference" in r["metrics"]: infer_pwr = r["metrics"]["inference"]["power_avg_w"]
        elif "inference_decode" in r["metrics"]: infer_pwr = r["metrics"]["inference_decode"]["power_avg_w"]
        elif "decode" in r["metrics"]: infer_pwr = r["metrics"]["decode"]["power_avg_w"]
        
        if train_pwr > 0: grouped_data[key]["train"].append(train_pwr)
        if infer_pwr > 0: grouped_data[key]["infer"].append(infer_pwr)

    plot_data = []
    for key, val in grouped_data.items():
        t_mean = np.mean(val["train"]) if val["train"] else 0
        t_std = np.std(val["train"]) if len(val["train"]) > 1 else 0
        i_mean = np.mean(val["infer"]) if val["infer"] else 0
        i_std = np.std(val["infer"]) if len(val["infer"]) > 1 else 0
        
        if t_mean == 0 and i_mean == 0: continue
        plot_data.append({
            "Label": key, "train_mean": t_mean, "train_std": t_std, "infer_mean": i_mean, "infer_std": i_std
        })
        
    if not plot_data: return None
    df = pd.DataFrame(plot_data).sort_values("Label") # Sort alphabetically by default
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Label"], y=df["train_mean"], name="Train Avg Power", marker_color="#3b82f6", opacity=0.8,
        error_y=dict(type='data', array=df["train_std"], visible=True)
    ))
    fig.add_trace(go.Scatter(
        x=df["Label"], y=df["infer_mean"], name="Inference Avg Power", mode='lines+markers',
        line=dict(color="#ef4444", width=3), marker=dict(size=10, symbol="circle"),
        error_y=dict(type='data', array=df["infer_std"], visible=True)
    ))
    fig.update_layout(
        title="<b>Train vs Inference Power Comparison (Avg over runs)</b>",
        xaxis_title="Condition", yaxis_title="Avg Power (W)",
        height=450, margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"), template="plotly_white",
        xaxis_tickangle=-15
    )
    return fig

def main():
    st.sidebar.title("⚡ AI Power Analysis")
    category = st.sidebar.radio("Category", ["Vision (GPU)", "LLM"])
    
    if category == "Vision (GPU)":
        base_dir = GPU_EXP_DIR
        meta = GPU_EXPERIMENTS
    else:
        base_dir = LLM_EXP_DIR
        meta = LLM_EXPERIMENTS
        
    experiments = load_experiments(base_dir, meta)
    
    st.title(f"📊 실험 결과 포털 - {category}")
    st.markdown("""
    **AI Workload Power Optimization Experiment Results**
    
    이 대시보드는 다양한 AI 모델(Vision, LLM)의 학습 및 추론 단계에서 발생하는 GPU 전력 소비 패턴을 분석합니다.
    모델 크기, 배치 크기, 정밀도(Precision), 전력 제한(Power Capping) 등 다양한 변인이 전력 효율(Performance/Watt)에 미치는 영향을 실험적으로 검증했습니다.
    """)
    st.divider()
    
    for exp_id, exp_meta in meta.items():
        if exp_id not in experiments: continue
        data = experiments[exp_id]
        runs = data["runs"]
        
        # Sort by GroupKey to group repeats visually
        runs = sorted(runs, key=lambda x: x["parsed_info"]["GroupKey"])
        
        with st.expander(f"📌 {exp_meta['title']} ({len(runs)} runs)", expanded=False):
            st.markdown(f"<div class='exp-desc'>{exp_meta['desc']}</div>", unsafe_allow_html=True)
            
            # 1. Comparison Chart
            st.subheader("📊 Power Comparison")
            comp_fig = create_comparison_chart(runs, category)
            if comp_fig: st.plotly_chart(comp_fig, use_container_width=True)
            st.divider()

            # 2. Detailed Metrics Table
            st.subheader("📋 Detailed Metrics")
            table_data = []
            for r in runs:
                parsed = r["parsed_info"]
                phases = list(r["metrics"].keys())
                phase = None
                if category == "LLM":
                    if "inference_decode" in phases: phase = "inference_decode"
                    elif "inference" in phases: phase = "inference"
                    elif "decode" in phases: phase = "decode"
                else:
                    if "train" in phases: phase = "train"
                    elif "inference" in phases: phase = "inference"
                if not phase and phases: phase = phases[0]
                
                run_label = parsed["Model"]
                if "Repeats" in parsed: run_label += f" ({parsed['Repeats']})"
                
                row = { "Model/Run": run_label, "Condition": parsed["Condition"] }
                
                if phase and phase in r["metrics"]:
                    m = r["metrics"][phase]
                    row["Avg Power (W)"] = f"{m.get('power_avg_w', 0):.1f}"
                    row["Energy (J)"] = f"{m.get('energy_j', 0):.1f}"
                    row["Time (s)"] = f"{m.get('duration_s', 0):.1f}"

                if category == "LLM" and r["infer_metrics"]:
                    total_tokens = r["infer_metrics"].get("total_output_tokens", 0)
                    if phase and phase in r["metrics"]:
                        dur = r["metrics"][phase].get("duration_s", 0)
                        en = r["metrics"][phase].get("energy_j", 0)
                        if dur > 0: row["Tokens/s"] = f"{total_tokens / dur:.1f}"
                        if total_tokens > 0: row["J/Token"] = f"{en / total_tokens:.2f}"

                table_data.append(row)
            
            if table_data:
                st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
            
            st.divider()
            
            # 3. Power Profile Grid
            st.subheader("📈 Power Profiles")
            cols = st.columns(2)
            for i, r in enumerate(runs):
                with cols[i % 2]:
                    parsed = r["parsed_info"]
                    plot_label = f"{parsed['Model']} - {parsed['Condition']}"
                    if "Repeats" in parsed: plot_label += f" [{parsed['Repeats']}]"
                    
                    df = load_gpu_samples(r["gpu_samples"])
                    if not df.empty and "phase" in df.columns:
                        if len(df) > 3000: df = df.iloc[::2, :]
                        st.plotly_chart(create_phase_plot(df, plot_label), use_container_width=True, key=f"c_{exp_id}_{i}")
                        
                        relative_path = r["path"].relative_to(OUTPUTS_ROOT).as_posix() 
                        base_url = "https://huggingface.co/datasets/aeoxxian/Datacenter_train/resolve/main/outputs"
                        raw_url = f"{base_url}/{relative_path}/samples/gpu_samples.csv"
                        st.link_button("☁️ Download Raw (Direct)", raw_url)

if __name__ == "__main__":
    main()
