"""
Streamlit Dashboard — A/B Testing & Uplift Modeling Explorer
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Uplift Modeling Explorer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme styles ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; }

    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-card .value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #58a6ff;
        line-height: 1;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #8b949e;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .delta {
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .positive { color: #3fb950; }
    .negative { color: #f78166; }
    .neutral  { color: #ffa657; }

    h1 { color: #e6edf3 !important; font-weight: 700 !important; }
    h2 { color: #c9d1d9 !important; font-weight: 600 !important; border-bottom: 1px solid #21262d; padding-bottom: 8px; }
    h3 { color: #8b949e !important; font-weight: 500 !important; }

    .stSidebar { background-color: #161b22 !important; border-right: 1px solid #21262d; }
    .stSidebar [data-testid="stMarkdownContainer"] { color: #c9d1d9; }

    .insight-box {
        background: linear-gradient(135deg, #1a2332 0%, #162032 100%);
        border-left: 4px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.95rem;
        color: #c9d1d9;
        line-height: 1.6;
    }

    .stSlider > div { color: #c9d1d9; }
    div[data-testid="metric-container"] { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

PALETTE = ['#58a6ff', '#3fb950', '#f78166', '#d2a8ff', '#ffa657']

plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#0d1117',
    'axes.edgecolor': '#30363d',   'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e',      'ytick.color': '#8b949e',
    'text.color': '#e6edf3',       'grid.color': '#21262d',
    'grid.linestyle': '--',        'font.size': 11,
})


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data_path = ROOT / 'data' / 'synthetic_users_full.parquet'
    if data_path.exists():
        df = pd.read_parquet(data_path)
        # Encode device
        df['device_enc'] = df['device'].map({'mobile': 0, 'desktop': 1, 'tablet': 2})
        return df
    return None


@st.cache_data
def load_model():
    model_path = ROOT / 'models' / 'xlearner_final.joblib'
    if model_path.exists():
        import joblib
        return joblib.load(model_path)
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_qini_curve(y, treatment, uplift_score, resolution=100):
    df_q = pd.DataFrame({'y': y, 'T': treatment, 'score': uplift_score})
    df_q = df_q.sort_values('score', ascending=False).reset_index(drop=True)
    N = len(df_q)
    pcts = [0]; qini = [0.0]
    for pct in np.linspace(100 / resolution, 100, resolution):
        n_tar = max(1, int(N * pct / 100))
        s = df_q.iloc[:n_tar]
        nt = s['T'].sum(); nc = (1 - s['T']).sum()
        if nt == 0 or nc == 0:
            qini.append(qini[-1]); pcts.append(pct); continue
        qini.append(s.loc[s.T == 1, 'y'].sum() - s.loc[s.T == 0, 'y'].sum() * (nt / nc))
        pcts.append(pct)
    return np.array(pcts), np.array(qini)


def make_metric_card(value, label, delta=None, delta_type='positive'):
    delta_html = ''
    if delta is not None:
        delta_html = f'<div class="delta {delta_type}">{delta}</div>'
    return f"""
    <div class="metric-card">
        <div class="value">{value}</div>
        <div class="label">{label}</div>
        {delta_html}
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

df = load_data()
model = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🎯 Uplift Modeling Explorer")
st.markdown(
    "An interactive dashboard for exploring heterogeneous treatment effects, "
    "targeting efficiency, and the business value of uplift modeling over standard A/B testing."
)

if df is None:
    st.error(
        "⚠️ Data not found. Please run **01_data_simulation.ipynb** first to generate "
        "`data/synthetic_users_full.parquet`."
    )
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Filters")
    st.markdown("---")

    selected_devices = st.multiselect(
        "Device type",
        options=['mobile', 'desktop', 'tablet'],
        default=['mobile', 'desktop', 'tablet'],
    )

    tenure_range = st.slider(
        "Tenure (days)",
        min_value=0, max_value=1825, value=(0, 1825), step=30,
    )

    engagement_range = st.slider(
        "Engagement score",
        min_value=0.0, max_value=1.0, value=(0.0, 1.0), step=0.05,
    )

    st.markdown("---")
    st.markdown("## 🎛️ Targeting")

    targeting_pct = st.slider(
        "Target top X% of users by predicted uplift",
        min_value=5, max_value=100, value=20, step=5,
        help="Simulate targeting only the users with the highest predicted uplift score.",
    )

    use_model_scores = st.toggle(
        "Use model predictions (vs. true ITE)",
        value=True if model is not None else False,
        disabled=(model is None),
        help="If enabled, uses the trained X-Learner model. If disabled, uses true ITE (requires full dataset).",
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#8b949e'>Built with ❤️ using Streamlit, scikit-learn, and causalML techniques.</small>",
        unsafe_allow_html=True
    )

# ── Apply filters ─────────────────────────────────────────────────────────────
mask = (
    df['device'].isin(selected_devices) &
    df['tenure_days'].between(*tenure_range) &
    df['engagement_score'].between(*engagement_range)
)
df_filt = df[mask].copy()

if len(df_filt) < 100:
    st.warning("⚠️ Too few users match the current filters. Adjust the sidebar filters.")
    st.stop()

# ── Top KPI Row ───────────────────────────────────────────────────────────────
st.markdown("## 📊 Experiment Summary")

n_ctrl  = (df_filt['treatment'] == 0).sum()
n_treat = (df_filt['treatment'] == 1).sum()
conv_ctrl   = df_filt.loc[df_filt.treatment == 0, 'y_obs'].mean()
conv_treat  = df_filt.loc[df_filt.treatment == 1, 'y_obs'].mean()
lift_obs    = conv_treat - conv_ctrl
avg_true_ite = df_filt['ite_true'].mean()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(make_metric_card(f"{len(df_filt):,}", "Users in segment"), unsafe_allow_html=True)
with col2:
    st.markdown(make_metric_card(f"{conv_ctrl:.3f}", "Control conv. rate"), unsafe_allow_html=True)
with col3:
    st.markdown(make_metric_card(f"{conv_treat:.3f}", "Treatment conv. rate"), unsafe_allow_html=True)
with col4:
    delta_sign = 'positive' if lift_obs > 0 else 'negative'
    st.markdown(
        make_metric_card(f"{lift_obs*100:+.2f}pp", "Observed lift",
                         delta=f"ATE", delta_type=delta_sign),
        unsafe_allow_html=True
    )
with col5:
    true_sign = 'positive' if avg_true_ite > 0 else 'negative'
    st.markdown(
        make_metric_card(f"{avg_true_ite*100:+.2f}pp", "True ITE (mean)",
                         delta="Ground truth", delta_type=true_sign),
        unsafe_allow_html=True
    )

# ── ITE Distribution ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 📈 Treatment Effect Distribution")
col_left, col_right = st.columns([3, 2])

with col_left:
    fig, ax = plt.subplots(figsize=(8, 4))
    device_colors = {'mobile': PALETTE[0], 'desktop': PALETTE[1], 'tablet': PALETTE[2]}
    for dev in selected_devices:
        subset = df_filt[df_filt['device'] == dev]['ite_true']
        if len(subset) > 0:
            ax.hist(subset * 100, bins=50, alpha=0.6, color=device_colors.get(dev, PALETTE[4]),
                    label=dev.capitalize(), edgecolor='none', density=True)
    ax.axvline(avg_true_ite * 100, color=PALETTE[2], lw=2.5, linestyle='--',
               label=f'Mean ITE = {avg_true_ite*100:+.2f}pp')
    ax.axvline(0, color='#8b949e', lw=1)
    ax.set_xlabel('True ITE (pp)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Distribution of True Individual Treatment Effects', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_right:
    # Segment table
    seg_tbl = (
        df_filt.groupby('segment')
        .agg(n=('ite_true', 'count'), avg_ite=('ite_true', 'mean'))
        .sort_values('avg_ite', ascending=False)
        .assign(avg_ite=lambda x: (x.avg_ite * 100).round(3))
        .rename(columns={'n': 'Users', 'avg_ite': 'Avg True ITE (pp)'})
        .reset_index()
    )
    st.markdown("#### Segment Breakdown")
    st.dataframe(seg_tbl, use_container_width=True, hide_index=True)

    st.markdown(
        f'<div class="insight-box">💡 <b>Key insight:</b> The average treatment effect '
        f'({avg_true_ite*100:+.2f}pp) masks significant heterogeneity across segments. '
        f'Blanket rollout misses this nuance entirely.</div>',
        unsafe_allow_html=True
    )

# ── Qini Curve & Targeting ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🎯 Targeting Efficiency")

# Compute uplift scores
if use_model_scores and model is not None:
    FEATURE_COLS = ['age', 'tenure_days', 'device_enc', 'past_spend', 'engagement_score']
    scores = model.predict(df_filt[FEATURE_COLS].values)
    score_label = 'X-Learner (Model)'
else:
    scores = df_filt['ite_true'].values
    score_label = 'True ITE (Oracle)'

y_vals = df_filt['y_obs'].values
t_vals = df_filt['treatment'].values

pcts, qini = compute_qini_curve(y_vals, t_vals, scores)
pcts_rand, qini_rand = compute_qini_curve(y_vals, t_vals, np.random.default_rng(42).random(len(y_vals)))
pcts_perf, qini_perf = compute_qini_curve(y_vals, t_vals, df_filt['ite_true'].values)

col_q, col_stats = st.columns([3, 2])

with col_q:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pcts, qini, color=PALETTE[0], lw=2.5, label=f'{score_label}')
    ax.plot(pcts_rand, qini_rand, color='#8b949e', lw=1.5, linestyle='--', label='Random')
    ax.plot(pcts_perf, qini_perf, color='#ffd700', lw=2, linestyle='-.', label='Perfect (true ITE)')
    ax.axvline(targeting_pct, color=PALETTE[2], lw=1.5, linestyle=':', alpha=0.9,
               label=f'Threshold: {targeting_pct}%')
    ax.set_xlabel('Population targeted (%)', fontsize=11)
    ax.set_ylabel('Incremental conversions', fontsize=11)
    ax.set_title(f'Qini Curve — {score_label}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col_stats:
    # Compute what-if stats at the threshold
    sorted_idx = np.argsort(scores)[::-1]
    n_targeted  = int(len(sorted_idx) * targeting_pct / 100)
    top_idx     = sorted_idx[:n_targeted]

    total_ite   = df_filt['ite_true'].values.sum()
    targeted_ite = df_filt['ite_true'].values[top_idx].sum()
    pct_lift_captured = targeted_ite / total_ite * 100 if total_ite > 0 else 0
    efficiency   = pct_lift_captured / targeting_pct if targeting_pct > 0 else 0

    st.markdown("#### What-If: Targeted Rollout")
    st.markdown(f"""
    | Metric | Value |
    |---|---|
    | Users targeted | {n_targeted:,} ({targeting_pct}%) |
    | Lift captured | **{pct_lift_captured:.1f}%** of total |
    | Efficiency vs. random | **{efficiency:.2f}×** |
    | Users spared the feature | {len(scores) - n_targeted:,} |
    """)

    st.markdown(
        f'<div class="insight-box">💡 Targeting the top <b>{targeting_pct}%</b> of users '
        f'captures <b>{pct_lift_captured:.1f}%</b> of total lift at <b>{targeting_pct}%</b> of the cost — '
        f'<b>{efficiency:.2f}×</b> more efficient than a blanket rollout.</div>',
        unsafe_allow_html=True
    )

    # Gain summary across thresholds
    st.markdown("#### Gain at Different Thresholds")
    thresholds_data = []
    for pct in [10, 20, 30, 50, 100]:
        n = int(len(sorted_idx) * pct / 100)
        top = sorted_idx[:n]
        captured = df_filt['ite_true'].values[top].sum() / total_ite * 100 if total_ite > 0 else 0
        eff = captured / pct if pct > 0 else 0
        thresholds_data.append({'Target %': f'{pct}%', 'Lift captured': f'{captured:.1f}%', 'Efficiency': f'{eff:.2f}×'})
    st.dataframe(pd.DataFrame(thresholds_data), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#8b949e; font-size:0.85rem; padding:16px 0'>"
    "A/B Testing & Uplift Modeling Portfolio Project | "
    "Built with Python, scikit-learn, Streamlit"
    "</div>",
    unsafe_allow_html=True
)
