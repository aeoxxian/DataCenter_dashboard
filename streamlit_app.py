
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
from pathlib import Path

# Set page config
st.set_page_config(layout="wide", page_title="AI Power Experiment Dashboard", page_icon="⚡")

# Custom CSS
st.markdown("""
<style>
    .exp-header { font-size: 1.2rem; font-weight: 600; color: #1e293b; }
    .exp-desc { color: #64748b; font-size: 0.9rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# Constants
HYP_OPT_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = HYP_OPT_ROOT / "outputs"
GPU_EXP_DIR = OUTPUTS_ROOT / "gpu_power_experiment"
LLM_EXP_DIR = OUTPUTS_ROOT / "llm_power_experiment"

# ── Experiment Metadata (Copied from build_transfer_package.py) ────────────────
GPU_EXPERIMENTS = {
    "00_baseline_reference": {"title": "기준 실험 (Baseline)", "desc": "전력 제어 없이 표준 조건으로 실행한 기준 실험입니다.", "baseline_key": None},
    "00_smoke_test": {"title": "파이프라인 검증 (Smoke Test)", "desc": "측정 파이프라인이 정상 작동하는지 확인하는 빠른 테스트입니다.", "baseline_key": None},
    "01_batch_sweep_resnet50": {"title": "배치 크기 변화 실험", "desc": "배치 크기(16/64/128)에 따른 GPU 전력 소비 변화를 관찰합니다.", "baseline_key": "bs64"},
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
    "02_decode_length_32_128": {"title": "디코딩 길이 변화 실험", "desc": "출력 토큰 길이(32 vs 128)에 따른 인퍼런스 전력 소비를 비교합니다.", "baseline_key": "maxtok32"},
    "03_dataset_alpaca_mmlu_longbench": {"title": "LLM 데이터셋 비교", "desc": "Alpaca(Inst), MMLU(QA), LongBench(Doc) 데이터셋별 특성입니다.", "baseline_key": "alpaca"},
    "04_decode_length_32_256_512": {"title": "긴 시퀀스 디코딩 실험", "desc": "최대 512 토큰까지 디코딩 길이를 늘려가며 전력 패턴을 관찰합니다.", "baseline_key": "maxtok32"},
    "05_inference_pattern_fixed_var_burst": {"title": "LLM 인퍼런스 패턴 실험", "desc": "지속적 요청(Fixed)과 간헐적 요청(Burst)의 전력 효율을 비교합니다.", "baseline_key": "fixed"},
    "06_power_cap_200_240_280_320_360": {"title": "LLM 전력 제한 실험", "desc": "LLM 인퍼런스 시 Power Capping의 효과와 성능 저하를 분석합니다.", "baseline_key": "nocap"},
    "07_clock_lock_1005_1500_2100": {"title": "LLM 클럭 고정 실험", "desc": "GPU 클럭 고정이 LLM 토큰 생성 속도와 전력에 미치는 영향입니다.", "baseline_key": "nocap"},
    "08_precision_fp16_bf16": {"title": "LLM 정밀도 비교 (FP16 vs BF16)", "desc": "FP16과 BF16의 전력 효율 및 메모리 사용량을 비교합니다.", "baseline_key": "fp16"},
    "09_control_combo_cap_clock_ramp": {"title": "전력/클럭 복합 제어 실험", "desc": "Cap, Clock, Ramp-up 제어를 조합하여 최적의 설정을 탐색합니다.", "baseline_key": "nocap"},
    "10_precision_bf16_fp16_4bit": {"title": "LLM 양자화 실험 (4bit)", "desc": "4bit 양자화가 전력 소비와 응답 속도에 미치는 영향을 분석합니다.", "baseline_key": "bf16"},
    "11_train_infer_split": {"title": "학습 vs 인퍼런스 분리 실험", "desc": "LLM의 학습(Fine-tuning)과 인퍼런스 단계를 분리하여 측정합니다.", "baseline_key": None},
    "12_model_scaling_tokenpowerbench": {"title": "토큰당 전력 벤치마크", "desc": "다양한 모델의 Token당 에너지(J/Token) 효율을 벤치마킹합니다.", "baseline_key": "gpt2"}
}

# Output Phase Colors (More Vivid & Distinct)
PHASE_COLORS = {
    # Generic
    "idle": "#94a3b8",      # slate-400 (Base Idle)
    "idle_pre": "#cbd5e1",  # slate-300 (Light Idle)
    "idle_post": "#64748b", # slate-500 (Dark Idle)
    "idle_mid": "#9ca3af",  # gray-400
    "warmup": "#fbbf24",    # amber-400
    "train": "#ef4444",     # red-500
    "train_compute": "#b91c1c", # red-700 (Darker for compute)
    "val": "#3b82f6",       # blue-500
    "validation": "#3b82f6",# blue-500
    "test": "#10b981",      # emerald-500
    # LLM
    "prefill": "#f59e0b",   # amber-500
    "decode": "#6366f1",    # indigo-500 (Distinct from Blue)
    "inference": "#8b5cf6", # violet-500
    "inference_prefill": "#f59e0b",
    "inference_decode": "#6366f1",
    "inference_idle": "#94a3b8",
}

PHASE_LABELS = {
    "idle_pre": "Idle (Pre)",
    "idle_post": "Idle (Post)",
    "idle_mid": "Idle (Mid)",
    "prefill": "Prefill",
    "decode": "Decode",
    "inference": "Inference",
    "train": "Training",
    "train_compute": "Train (Compute)",
    "warmup": "Warmup",
    "val": "Validation",
    "validation": "Validation",
    "inference_prefill": "Prefill",
    "inference_decode": "Decode",
    "inference_idle": "Idle",
}

@st.cache_data
def load_experiments(base_dir, experiment_meta):
    """Scan experiment directory with strict folder name matching."""
    experiments = {}
    if not base_dir.exists():
        return experiments

    # Iterate through metadata keys to ensure order and existence
    for exp_id, meta in experiment_meta.items():
        entry = base_dir / exp_id
        if entry.exists() and entry.is_dir():
            experiments[exp_id] = {
                "meta": meta,
                "path": entry,
                "runs": []
            }
            # Scan runs - Assume standard structure
            for run_dir in sorted(entry.rglob("config.json")):
                run_path = run_dir.parent
                
                # 1. Load Power Metrics
                metrics_path = run_path / "samples" / "phase_power_summary.csv"
                if not metrics_path.exists():
                    metrics_path = run_path / "phase_power_summary.csv"
                
                metrics = {}
                if metrics_path.exists():
                    try:
                        df = pd.read_csv(metrics_path)
                        # Fix column names if needed or expect 'power_avg_w'
                        if "power_avg_w" in df.columns:
                            metrics = df.set_index("phase").to_dict(orient="index")
                    except Exception:
                        pass
                
                # 2. Load Inference Metrics
                infer_metrics = {}
                infer_path = run_path / "samples" / "inference_results.csv"
                if infer_path.exists():
                    try:
                        df_infer = pd.read_csv(infer_path)
                        if "output_tokens" in df_infer.columns:
                            infer_metrics["total_output_tokens"] = df_infer["output_tokens"].sum()
                    except Exception:
                        pass

                # 3. Load Config
                try:
                    with open(run_dir) as f:
                        config = json.load(f)
                except:
                    config = {}

                # Create Label (Full Name)
                label = run_path.name
                # if len(label) > 50: label = label[:48] + "..." # Removed Truncation check

                # Find GPU samples path (Prioritize Lite version for cloud deployment)
                gpu_samples_path = run_path / "gpu_samples_lite.csv"
                if not gpu_samples_path.exists():
                    gpu_samples_path = run_path / "samples" / "gpu_samples.csv"
                if not gpu_samples_path.exists():
                    gpu_samples_path = run_path / "gpu_samples.csv"

                experiments[exp_id]["runs"].append({
                    "name": run_path.name,
                    "label": label,
                    "path": run_path,
                    "metrics": metrics,
                    "infer_metrics": infer_metrics,
                    "config": config,
                    "gpu_samples": gpu_samples_path
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
    """Create a Plotly figure with phase-based coloring."""
    fig = go.Figure()
    
    if df.empty or "phase" not in df.columns:
        return fig
        
    t_start = df["timestamp_ms"].min()
    df["time"] = (df["timestamp_ms"] - t_start) / 1000.0
    
    # Keep original resolution for better visual or slight downsample
    # Phase highlighting works better with all points or minimal downsample.
    
    df['phase_group'] = (df['phase'] != df['phase'].shift()).cumsum()
    added_legends = set()
    
    for _, group in df.groupby('phase_group'):
        phase = group['phase'].iloc[0]
        color = PHASE_COLORS.get(phase, "#000000")
        name = PHASE_LABELS.get(phase, phase)
        
        show_legend = name not in added_legends
        if show_legend:
            added_legends.add(name)
            
        fig.add_trace(go.Scatter(
            x=group["time"], 
            y=group["power_w"],
            mode='lines',
            name=name,
            line=dict(color=color, width=2.5), # Thicker line
            legendgroup=name,
            showlegend=show_legend,
            hoverinfo='x+y+name'
        ))
        
    fig.update_layout(
        title=dict(text=label, font=dict(size=16, color="#1e293b", family="sans-serif")),
        height=500, # Increased height
        margin=dict(l=20, r=20, t=50, b=50),
        xaxis=dict(title="Time (s)", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(title="Power (W)", showgrid=True, gridcolor="#f1f5f9"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h", 
            y=-0.25, # Lower legend
            xanchor="center", x=0.5, 
            font=dict(size=12, color="#1e293b"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e2e8f0",
            borderwidth=1
        ),
        hovermode="x unified"
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
    st.markdown("모든 실험 결과를 한눈에 비교하고 분석할 수 있습니다. 각 항목을 클릭하여 상세 내용을 확인하세요.")
    
    for exp_id, exp_meta in meta.items():
        if exp_id not in experiments:
            continue
            
        data = experiments[exp_id]
        runs = data["runs"]
        baseline_key = exp_meta.get("baseline_key")
        
        # Determine Baseline run
        base_run = None
        if baseline_key:
            for r in runs:
                if baseline_key in r["name"]:
                    base_run = r
                    break
        
        # Expanded by default? Users might want to see overview first.
        # User asked for "show plots automatically", implying when they open the expander.
        # So we keep expander collapsed by default (to save space), but once opened, plots are there.
        with st.expander(f"{exp_meta['title']} ({len(runs)} runs)", expanded=False):
            st.markdown(f"<div class='exp-desc'>{exp_meta['desc']}</div>", unsafe_allow_html=True)
            
            # --- Metrics Table ---
            table_data = []
            for r in runs:
                phases = list(r["metrics"].keys())
                phase = None
                if category == "LLM":
                    if "inference_decode" in phases: phase = "inference_decode"
                    elif "inference" in phases: phase = "inference"
                else:
                    if "train" in phases: phase = "train"
                    elif "inference" in phases: phase = "inference"
                
                if not phase and phases: phase = phases[0]
                
                row = {"Run": r["label"]}
                if phase and phase in r["metrics"]:
                    m = r["metrics"][phase]
                    pwr = m.get("power_avg_w", 0)
                    eng = m.get("energy_j", 0)
                    time = m.get("duration_s", 0)
                    row["Power (W)"] = f"{pwr:.1f}"
                    row["Energy (J)"] = f"{eng:.1f}"
                    row["Time (s)"] = f"{time:.1f}"

                    if base_run and base_run != r and phase in base_run["metrics"]:
                        bm = base_run["metrics"][phase]
                        d_pwr = calculate_delta(pwr, bm.get("power_avg_w", 0))
                        d_eng = calculate_delta(eng, bm.get("energy_j", 0))
                        d_time = calculate_delta(time, bm.get("duration_s", 0))
                        if d_pwr: row["Power (W)"] += f" ({d_pwr:+.1f}%)"
                        if d_eng: row["Energy (J)"] += f" ({d_eng:+.1f}%)"
                        if d_time: row["Time (s)"] += f" ({d_time:+.1f}%)"
                
                if category == "LLM" and r["infer_metrics"] and phase in r["metrics"]:
                    calc_duration = r["metrics"][phase].get("duration_s", 0)
                    calc_energy = r["metrics"][phase].get("energy_j", 0)
                    total_tokens = r["infer_metrics"].get("total_output_tokens", 0)
                    if calc_duration > 0:
                        tps = total_tokens / calc_duration
                        row["Tok/s"] = f"{tps:.1f}"
                    if total_tokens > 0:
                        jpt = calc_energy / total_tokens
                        row["J/Tok"] = f"{jpt:.2f}"

                table_data.append(row)
            
            if table_data:
                st.dataframe(pd.DataFrame(table_data).set_index("Run"), use_container_width=True)
            
            # --- Plots Grid (Always Visible) ---
            st.markdown("##### 📈 Power Profile")
            cols = st.columns(2)
            for i, r in enumerate(runs):
                with cols[i % 2]:
                    df = load_gpu_samples(r["gpu_samples"])
                    if not df.empty and "phase" in df.columns:
                        if len(df) > 3000: # Slightly less aggressive downsampling
                            df = df.iloc[::2, :]
                        
                        fig = create_phase_plot(df, r["label"])
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{exp_id}_{i}")

if __name__ == "__main__":
    main()
