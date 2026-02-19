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
    "01_batch_sweep_resnet50": {"title": "배치 크기 변화 실험", "desc": "배치 크기(16/64/128)에 따른 GPU 전력 소비 변화를 관찰합니다.", "baseline_key": "bs64"},
    "02_operator_control": {"title": "운영자 제어 시뮬레이션", "desc": "비균일 추론 요청 패턴에서의 전력 프로파일을 관찰합니다.", "baseline_key": None},
    "03_control_cap_sweep": {"title": "전력 상한 (Power Cap) 실험", "desc": "nvidia-smi 전력 제한(345W/460W/575W)이 성능과 전력에 미치는 영향을 테스트합니다.", "baseline_key": "cap575W"},
    "04_control_ramp": {"title": "단계적 전력 램프 실험", "desc": "전력 제한을 단계적으로 변화시켜 GPU 적응 동작을 관찰합니다.", "baseline_key": "nocap"},
    "05_pattern_fixed_var_burst": {"title": "추론 패턴 비교 실험", "desc": "고정/가변/버스트 세 가지 추론 스케줄링 패턴의 전력 영향을 비교합니다.", "baseline_key": "fixed"},
    "06_model_scaling_image": {"title": "모델 아키텍처별 전력 비교", "desc": "10개 비전 모델 아키텍처가 생성하는 서로 다른 전력 시그니처를 비교합니다.", "baseline_key": "resnet50"},
    "07_dataset_cifar10_cifar100_imagenet": {"title": "데이터셋별 전력 비교", "desc": "데이터셋(이미지 해상도, 클래스 수)이 전력 프로파일에 미치는 영향을 테스트합니다.", "baseline_key": "cifar10"},
    "08_train_modes_fixed_automl": {"title": "학습 모드 비교", "desc": "고정 SGD 학습 vs Optuna 기반 자동 하이퍼파라미터 탐색의 전력 비교입니다.", "baseline_key": "fixed"},
    "10_checkpoint_iteration_observe": {"title": "체크포인트 I/O 관찰", "desc": "주기적 모델 저장이 만드는 전력 스파이크(GPU 연산 중단 → 디스크 I/O)를 관찰합니다.", "baseline_key": None},
    "11_control_clock_sweep": {"title": "클럭 주파수 고정 실험", "desc": "GPU SM 클럭을 특정 주파수(1005/1500/2100MHz)로 고정했을 때의 전력 변화를 테스트합니다.", "baseline_key": "clk2100MHz"}
}

LLM_EXPERIMENTS = {
    "01_model_scaling_initial": {"title": "LLM 모델 스케일링 (초기)", "desc": "GPT-2, Qwen3-4B, Mistral-7B, Llama3.1-8B 네 종류의 LLM 전력 프로파일을 비교합니다.", "baseline_key": "qwen3-4b"},
    "02_decode_length_32_128": {"title": "디코드 길이 비교 (32 vs 128)", "desc": "최대 토큰 생성 길이(32/128)에 따른 전력 변화를 관찰합니다.", "baseline_key": "maxtok32"},
    "03_dataset_alpaca_mmlu_longbench": {"title": "LLM 데이터셋 비교", "desc": "Alpaca, MMLU-Pro, LongBench 데이터셋별 전력 프로파일을 비교합니다.", "baseline_key": "alpaca"},
    "04_decode_length_32_256_512": {"title": "디코드 길이 확장 비교 (32/256/512)", "desc": "더 넓은 범위의 최대 생성 토큰 수(32/256/512)별 전력 변화를 관찰합니다.", "baseline_key": "maxtok32"},
    "05_inference_pattern_fixed_var_burst": {"title": "LLM 추론 패턴 비교", "desc": "고정/가변/버스트/동시 버스트 네 가지 추론 패턴이 전력과 처리량에 미치는 영향을 비교합니다.", "baseline_key": "fixed"},
    "06_power_cap_200_240_280_320_360": {"title": "LLM 전력 상한 (Cap) 실험", "desc": "전력 제한(200~575W)이 LLM 추론 성능(tokens/s)과 효율(J/token)에 미치는 영향을 테스트합니다.", "baseline_key": "cap360W"},
    "07_clock_lock_1005_1500_2100": {"title": "LLM 클럭 고정 실험", "desc": "GPU 클럭 고정(1005/1500/2100MHz)이 LLM 추론 성능에 미치는 영향을 테스트합니다.", "baseline_key": "clk2100MHz"},
    "08_ramp_rate": {"title": "LLM 전력 램프 실험", "desc": "단계적 전력 변화가 LLM 추론 중 전력 소비에 미치는 영향을 관찰합니다.", "baseline_key": None},
    "09_control_combo_cap_clock_ramp": {"title": "복합 제어 실험 (Cap + Clock + Ramp)", "desc": "전력 상한, 클럭 고정, 램프를 조합한 복합 제어의 효과를 테스트합니다.", "baseline_key": None},
    "10_precision_bf16_fp16_4bit": {"title": "정밀도별 전력 비교 (BF16/FP16/4bit)", "desc": "연산 정밀도(BF16/FP16/4bit 양자화)에 따른 전력 및 성능 변화를 비교합니다.", "baseline_key": "fp16"},
    "11_train_infer_split": {"title": "학습/추론 분리 실험", "desc": "학습만, 추론만, 전체 사이클의 전력 프로파일 차이를 비교합니다.", "baseline_key": "full_cycle"},
    "12_model_scaling_tokenpowerbench": {"title": "LLM 모델별 토큰-전력 벤치마크", "desc": "GPT-2, Qwen3-4B, Mistral-7B, Llama3.1-8B의 tokens/s 및 J/token을 벤치마크합니다.", "baseline_key": "qwen3-4b"}
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
    실험 ID(context_exp_id)에 따라 가장 중요한 변인(Variable)을 추출하여 라벨링합니다.
    파일명, config.json 내용을 종합적으로 활용합니다.
    """
    info = {
        "Model": "Unknown",
        "Condition": "",
        "GroupKey": "",
        "Repeats": []
    }
    
    # --- 1) 모델명 추출 ---
    name_lower = run_name.lower()
    if "resnet50" in name_lower: info["Model"] = "ResNet50"
    elif "resnet18" in name_lower: info["Model"] = "ResNet18"
    elif "vgg16" in name_lower: info["Model"] = "VGG16"
    elif "densenet121" in name_lower: info["Model"] = "DenseNet121"
    elif "mobilenetv2" in name_lower: info["Model"] = "MobileNetV2"
    elif "efficientnet_b0" in name_lower: info["Model"] = "EfficientNet-B0"
    elif "convnext_tiny" in name_lower: info["Model"] = "ConvNeXt-Tiny"
    elif "resnext50" in name_lower: info["Model"] = "ResNeXt50"
    elif "swin_t" in name_lower: info["Model"] = "Swin-T"
    elif "vit_b_16" in name_lower: info["Model"] = "ViT-B/16"
    elif "qwen3-4b" in name_lower: info["Model"] = "Qwen3-4B"
    elif "llama3.1-8b" in name_lower: info["Model"] = "Llama3.1-8B"
    elif "mistral-7b" in name_lower: info["Model"] = "Mistral-7B"
    elif "gpt2" in name_lower: info["Model"] = "GPT-2"
    else: info["Model"] = run_name.split('_')[0]

    # --- 2) 실험 컨텍스트 기반 조건 추출 ---
    conditions = []
    
    # 헬퍼 함수: Config 값 우선 확인, 없으면 파일명에서 검색
    def get_val(key, default=None):
        return config.get(key, default) if config else default
        
    def check_in_name(keywords):
        return any(k in name_lower for k in keywords)

    if context_exp_id:
        # === GPU Experiments ===
        if "01_batch_sweep" in context_exp_id:
            bs = get_val("batch_size")
            if bs: conditions.append(f"BS {bs}")
            elif "bs16" in name_lower: conditions.append("BS 16")
            elif "bs64" in name_lower: conditions.append("BS 64")
            elif "bs128" in name_lower: conditions.append("BS 128")
            
        elif "02_operator_control" in context_exp_id:
            conditions.append("Variable Request Interval")
            
        elif "03_control_cap_sweep" in context_exp_id:
            cap = get_val("power_cap")
            if cap: conditions.append(f"Cap {cap}W")
            elif "cap345w" in name_lower: conditions.append("Cap 345W")
            elif "cap460w" in name_lower: conditions.append("Cap 460W")
            elif "cap575w" in name_lower: conditions.append("Cap 575W")
            elif "nocap" in name_lower: conditions.append("No Cap")

        elif "04_control_ramp" in context_exp_id:
            conditions.append("Ramp Active")

        elif "05_pattern" in context_exp_id: # 05_pattern_fixed_var_burst both GPU and LLM
            if "burst" in name_lower and "concurrent" not in name_lower: conditions.append("Burst")
            elif "concurrent_burst" in name_lower: conditions.append("Concurrent Burst")
            elif "variable" in name_lower: conditions.append("Variable")
            elif "fixed" in name_lower: conditions.append("Fixed")
            
        elif "06_model_scaling" in context_exp_id: # GPU #06
            # 모델 자체가 변인이므로 추가 조건 불필요, 하지만 Config 확인
            pass

        elif "07_dataset" in context_exp_id:
            ds = get_val("dataset")
            if ds: conditions.append(ds)
            elif "cifar100" in name_lower: conditions.append("CIFAR-100")
            elif "imagenet" in name_lower: conditions.append("ImageNet")
            elif "cifar10" in name_lower: conditions.append("CIFAR-10")

        elif "08_train_modes" in context_exp_id and "automl" in name_lower:
            conditions.append("AutoML")
            
        elif "10_checkpoint" in context_exp_id:
             conditions.append("Checkpoint Active")
             
        elif "11_control_clock" in context_exp_id: # GPU #11
            if "clk1005mhz" in name_lower: conditions.append("Clock 1005MHz")
            elif "clk1500mhz" in name_lower: conditions.append("Clock 1500MHz")
            elif "clk2100mhz" in name_lower: conditions.append("Clock 2100MHz")

        # === LLM Experiments ===
        elif "01_model_scaling" in context_exp_id: # LLM #01
             # 모델 자체가 변인
             pass
             
        elif "02_decode_length" in context_exp_id or "04_decode_length" in context_exp_id:
            # Check for maxtok explicitly
            if "maxtok32" in name_lower: conditions.append("MaxTok 32")
            elif "maxtok128" in name_lower: conditions.append("MaxTok 128")
            elif "maxtok256" in name_lower: conditions.append("MaxTok 256")
            elif "maxtok512" in name_lower: conditions.append("MaxTok 512")
            else:
                tok = get_val("gen_max_new_tokens")
                if tok: conditions.append(f"MaxTok {tok}")
        
        elif "03_dataset" in context_exp_id: # LLM #03
            if "alpaca" in name_lower: conditions.append("Alpaca")
            elif "mmlu-pro" in name_lower: conditions.append("MMLU-Pro")
            elif "longbench" in name_lower: conditions.append("LongBench")
            
        elif "05_inference_pattern" in context_exp_id: # LLM #05
            if "concurrent_burst" in name_lower: conditions.append("Concurrent Burst")
            elif "burst" in name_lower: conditions.append("Burst")
            elif "variable" in name_lower: conditions.append("Variable")
            else: conditions.append("Fixed")
            
        elif "06_power_cap" in context_exp_id: # LLM #06
             # Extract Cap from name or config
             found_cap = False
             match = re.search(r"cap(\d+)w", name_lower)
             if match: 
                 conditions.append(f"Cap {match.group(1)}W")
                 found_cap = True
             if not found_cap:
                 cap = get_val("power_cap")
                 if cap: conditions.append(f"Cap {cap}W")
        
        elif "07_clock_lock" in context_exp_id: # LLM #07
            if "clk1005mhz" in name_lower: conditions.append("Clock 1005MHz")
            elif "clk1500mhz" in name_lower: conditions.append("Clock 1500MHz")
            elif "clk2100mhz" in name_lower: conditions.append("Clock 2100MHz")
            
        elif "08_ramp_rate" in context_exp_id: # LLM #08
            conditions.append("Ramp")
            
        elif "09_control_combo" in context_exp_id:
            # 복합 조건 파싱
            combo = []
            if "cap" in name_lower:
                match = re.search(r"cap(\d+)w", name_lower)
                if match: combo.append(f"Cap {match.group(1)}W")
            if "clk" in name_lower:
                match = re.search(r"clk(\d+)mhz", name_lower)
                if match: combo.append(f"Clock {match.group(1)}MHz")
            if "ramp" in name_lower: combo.append("Ramp")
            if combo: conditions.append(" + ".join(combo))
            
        elif "10_precision" in context_exp_id: # LLM #10
            if "4bit" in name_lower: conditions.append("4-bit")
            elif "fp16" in name_lower: conditions.append("FP16")
            elif "bf16" in name_lower: conditions.append("BF16")
            elif "nocap" in name_lower: conditions.append("BF16 (Default)") # 보통 Default가 BF16
            
        elif "11_train_infer_split" in context_exp_id:
            if "train_only" in name_lower: conditions.append("Train Only")
            elif "infer_only" in name_lower: conditions.append("Inference Only")
            elif "full_cycle" in name_lower or "nocap" in name_lower: conditions.append("Full Cycle")
            
        elif "12_model_scaling" in context_exp_id: # LLM #12
            pass

    # --- 3) 조건 조합 및 클린업 ---
    # 중복 제거 및 정렬
    unique_conditions = sorted(list(set(conditions)))
    if unique_conditions:
        info["Condition"] = ", ".join(unique_conditions)
    else:
        # 조건이 특별히 없으면 Standard 또는 Baseline 표시
        if "nocap" in name_lower and "fixed" in name_lower: info["Condition"] = "Standard (Fixed/NoCap)"
        else: info["Condition"] = "Standard"

    # --- 4) 반복 실험(Repeat) 처리 ---
    # 폴더 구조상 repeats/r01, r02 등인지 확인
    parts = run_path.parts
    is_repeat = False
    repeat_idx = ""
    for part in parts:
        if re.match(r'^r\d+$', part):
            is_repeat = True
            repeat_idx = part
            break
            
    info["GroupKey"] = f"{info['Model']} ({info['Condition']})"
    if info["Condition"] == "Standard": info["GroupKey"] = info["Model"]
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
