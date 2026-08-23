import json
import io
import os
import csv
from datetime import datetime, timezone
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
from groq import Groq
from dotenv import load_dotenv
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

load_dotenv()  # reads GROQ_API_KEY from a local .env file, if present

st.set_page_config(page_title="FinSight AI", page_icon="💰", layout="wide")

# ---------------------------------------------------------------------
# Theme — deep-navy "risk terminal" look: Space Grotesk for headings,
# Inter for body copy, JetBrains Mono for every number (precision,
# recall, dollar figures) so metrics read like live instrument data.
# ---------------------------------------------------------------------
PANEL_BG = "#121A2E"
GRID_COLOR = "#232D48"
TEXT_COLOR = "#E8ECF4"
MUTED = "#8B96AD"
ACCENT = "#2DD4BF"
ACCENT_SOFT = "#5EEAD4"
DANGER = "#FB7185"
SAFE = "#34D399"
BG = "#0A0E1A"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600;700&display=swap');

:root {{
    --bg: {BG}; --panel: {PANEL_BG}; --panel-border: {GRID_COLOR};
    --accent: {ACCENT}; --danger: {DANGER}; --safe: {SAFE};
    --text: {TEXT_COLOR}; --muted: {MUTED};
}}

.stApp {{
    background: radial-gradient(circle at 15% 0%, #101833 0%, var(--bg) 55%);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}}
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
.block-container {{ padding-top: 1.2rem !important; max-width: 1180px; }}

@keyframes fadeInUp {{ from {{opacity:0; transform: translateY(10px);}} to {{opacity:1; transform: translateY(0);}} }}

h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }}
h3 {{ font-weight: 600 !important; color: var(--text) !important; }}
p, .stMarkdown, .stCaption, label {{ font-family: 'Inter', sans-serif !important; }}

/* ---------------- TOP BAR ---------------- */
.topbar {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 4px 18px; margin-bottom: 4px;
    border-bottom: 1px solid rgba(35,45,72,0.6);
}}
.topbar-brand {{
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.1rem; color: var(--text);
    display: flex; align-items: center; gap: 8px;
}}
.topbar-brand .mark {{
    width: 26px; height: 26px; border-radius: 8px;
    background: linear-gradient(135deg, var(--accent), #0f9c8c);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; box-shadow: 0 0 16px rgba(45,212,191,0.35);
}}
.topbar-links {{ display: flex; align-items: center; gap: 18px; }}
.topbar-links a {{
    font-family: 'Inter', sans-serif; font-size: 0.86rem; color: var(--muted) !important;
    text-decoration: none !important; transition: color .15s ease;
}}
.topbar-links a:hover {{ color: var(--text) !important; }}
.status-pill {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: var(--safe) !important;
    border: 1px solid rgba(52,211,153,0.35); border-radius: 999px; padding: 4px 11px;
    display: flex; align-items: center; gap: 6px;
}}
.status-pill .pulse {{
    width: 6px; height: 6px; border-radius: 50%; background: var(--safe);
    box-shadow: 0 0 0 3px rgba(52,211,153,0.22);
}}

/* ---------------- HERO ---------------- */
.hero {{
    position: relative; overflow: hidden; animation: fadeInUp .5s ease;
    background: linear-gradient(135deg, rgba(45,212,191,0.14), rgba(18,26,46,0) 55%), var(--panel);
    border: 1px solid var(--panel-border); border-radius: 22px;
    padding: 40px 44px 30px; margin-bottom: 22px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.35);
}}
.hero::before {{
    content: ""; position: absolute; top: -120px; right: -120px; width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(45,212,191,0.35), transparent 70%);
    border-radius: 50%; pointer-events: none;
}}
.hero-eyebrow {{
    font-family: 'JetBrains Mono', monospace; color: var(--accent); font-size: 0.76rem;
    letter-spacing: 0.1em; text-transform: uppercase; display: flex; align-items: center; gap: 8px;
    margin-bottom: 12px;
}}
.hero-eyebrow .dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--safe); box-shadow: 0 0 0 4px rgba(52,211,153,0.18); }}
.hero h1 {{ font-size: 2.5rem !important; font-weight: 700 !important; margin: 0 0 8px 0 !important; padding: 0 !important; }}
.hero-sub {{ color: var(--muted); font-size: 1.02rem; max-width: 740px; line-height: 1.6; margin-bottom: 22px; }}
.hero-sub a {{ color: var(--accent); text-decoration: none; border-bottom: 1px dashed rgba(45,212,191,0.5); }}

.badge-row {{ display: flex; flex-wrap: wrap; gap: 9px; margin-bottom: 24px; }}
.badge {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; padding: 6px 12px; border-radius: 999px;
    border: 1px solid var(--panel-border); background: rgba(255,255,255,0.02); color: var(--muted);
    transition: border-color .15s ease, color .15s ease;
}}
.badge:hover {{ border-color: rgba(45,212,191,0.4); color: var(--text); }}
.badge.live {{ color: var(--safe); border-color: rgba(52,211,153,0.35); }}

.kpi-strip {{ display: flex; gap: 12px; flex-wrap: wrap; }}
.kpi {{
    flex: 1 1 145px; background: rgba(255,255,255,0.03); border: 1px solid var(--panel-border);
    border-radius: 14px; padding: 13px 16px; transition: transform .15s ease, border-color .15s ease;
}}
.kpi:hover {{ transform: translateY(-3px); border-color: rgba(45,212,191,0.4); }}
.kpi-label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-bottom: 4px; }}
.kpi-value {{
    font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; color: var(--accent);
    display: inline-block; animation: kpiPop .55s cubic-bezier(.2,.8,.3,1.2) backwards;
}}
.kpi-value.warn {{ color: var(--danger); }}
@keyframes kpiPop {{ from {{ opacity: 0; transform: scale(.7) translateY(4px); }} to {{ opacity: 1; transform: scale(1) translateY(0); }} }}
.kpi:nth-child(1) .kpi-value {{ animation-delay: .05s; }}
.kpi:nth-child(2) .kpi-value {{ animation-delay: .12s; }}
.kpi:nth-child(3) .kpi-value {{ animation-delay: .19s; }}
.kpi:nth-child(4) .kpi-value {{ animation-delay: .26s; }}
.kpi:nth-child(5) .kpi-value {{ animation-delay: .33s; }}

/* ---------------- CARD SURFACES (wraps every chart/section) ---------------- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: var(--panel) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.25);
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    animation: fadeInUp .45s ease;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    border-color: rgba(45,212,191,0.32) !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    transform: translateY(-2px);
}}
.card-header {{ display: flex; align-items: center; gap: 12px; padding: 6px 6px 2px; margin-bottom: 4px; }}
.card-icon {{
    width: 34px; height: 34px; border-radius: 10px; background: rgba(45,212,191,0.12);
    display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0;
}}
.card-title {{ font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.03rem; color: var(--text); line-height: 1.2; }}
.card-sub {{ font-size: 0.8rem; color: var(--muted); margin-top: 1px; }}

/* Metric cards */
div[data-testid="stMetric"] {{
    background: rgba(255,255,255,0.02); border: 1px solid var(--panel-border);
    border-radius: 12px; padding: 14px 16px 10px;
    transition: transform .15s ease, border-color .15s ease;
}}
div[data-testid="stMetric"]:hover {{ transform: translateY(-2px); border-color: rgba(45,212,191,0.32); }}
div[data-testid="stMetricLabel"] {{
    font-family: 'Inter', sans-serif !important; color: var(--muted) !important; font-size: 0.83rem !important;
    text-transform: uppercase; letter-spacing: 0.04em;
}}
div[data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', monospace !important; color: var(--accent) !important; font-weight: 700 !important; }}

/* ---------------- Pill-style Tabs ---------------- */
div[data-baseweb="tab-list"] {{
    background: rgba(255,255,255,0.03); border: 1px solid var(--panel-border);
    border-radius: 13px; padding: 5px; gap: 4px; width: fit-content;
}}
button[data-baseweb="tab"] {{
    font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important; font-size: 0.92rem !important;
    color: var(--muted) !important; border-radius: 9px !important; padding: 8px 16px !important;
    transition: all .15s ease;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    background: var(--accent) !important; color: #05231F !important;
}}
div[data-baseweb="tab-highlight"] {{ display: none !important; }}
div[data-baseweb="tab-border"] {{ display: none !important; }}
.stTabs {{ margin-bottom: 6px; }}

/* Sliders / buttons */
div[data-testid="stSlider"] > div > div > div > div {{ background-color: var(--accent) !important; }}
button[kind="primary"] {{
    background: var(--accent) !important; color: #05231F !important; border: none !important;
    font-weight: 600 !important; border-radius: 8px !important;
}}
button[kind="primary"]:hover {{ background: #5EEAD4 !important; }}

/* Alerts / Expander / Dataframe / Captions / Links */
div[data-testid="stAlert"] {{ border-radius: 10px; font-family: 'Inter', sans-serif; }}
details {{ background: rgba(255,255,255,0.02); border: 1px solid var(--panel-border) !important; border-radius: 10px; }}
div[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; border: 1px solid var(--panel-border); }}
[data-testid="stCaptionContainer"] {{ color: var(--muted) !important; }}
a {{ color: var(--accent) !important; }}

/* ---------------- Footer ---------------- */
.app-footer {{
    margin-top: 34px; padding-top: 18px; border-top: 1px solid var(--panel-border);
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
    color: var(--muted); font-size: 0.82rem;
}}
.app-footer .badges {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.app-footer .badges span {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; padding: 4px 10px; border-radius: 999px;
    border: 1px solid var(--panel-border); color: var(--muted);
}}
</style>
""", unsafe_allow_html=True)


def card_header(icon, title, subtitle=None):
    sub_html = f'<div class="card-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="card-header">
      <div class="card-icon">{icon}</div>
      <div>
        <div class="card-title">{title}</div>
        {sub_html}
      </div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv('data/transactions_sample.csv')


@st.cache_data
def load_metrics():
    with open('models/metrics.json') as f:
        return json.load(f)


@st.cache_data
def load_test_predictions():
    return pd.read_csv('models/test_predictions.csv')


@st.cache_resource
def get_shap_explainer(_model):
    # TreeExplainer works directly on the trained forest — no retraining,
    # no surrogate model. Cached as a resource so it's built once per session.
    return shap.TreeExplainer(_model)


@st.cache_resource
def get_llm_client():
    # Groq offers a free, no-credit-card API tier — reads the key from an
    # environment variable, never hardcode it here.
    # On Render: Dashboard → your service → Environment → add GROQ_API_KEY.
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def explain_with_llm(shap_df, probability, prediction):
    """Turn the top SHAP drivers into a short, analyst-readable explanation.
    This is a communication layer only — it never changes the model's
    decision, it just narrates the SHAP values that already drove it."""
    client = get_llm_client()
    if client is None:
        return None, "GROQ_API_KEY not set — LLM explanation unavailable."

    top = shap_df.tail(3).iloc[::-1]  # 3 biggest |SHAP| drivers, biggest first
    feature_summary = "\n".join(
        f"- {row['feature']}: value={row['value']:.2f} "
        f"({'pushes toward fraud' if row['shap'] > 0 else 'pushes toward genuine'}, "
        f"strength {abs(row['shap']):.3f})"
        for _, row in top.iterrows()
    )
    prompt = (
        f"A fraud-detection model scored a transaction as "
        f"{'FRAUD' if prediction == 1 else 'GENUINE'} with a "
        f"{probability*100:.1f}% fraud probability.\n\n"
        f"Top contributing factors (from SHAP attribution):\n{feature_summary}\n\n"
        "In 2-3 short sentences, explain this decision to a bank risk analyst in "
        "plain English. Be concrete about the direction of each factor. Do not "
        "invent information beyond what's given above."
    )
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, f"LLM explanation failed: {e}"


def log_prediction(amt, category, gender, hour, probability, prediction, source="live"):
    """Lightweight audit trail: one row per prediction. Note — on Render's
    free tier the filesystem is ephemeral (resets on redeploy/restart), so
    treat this as a demo-grade audit trail, not a production log store."""
    try:
        os.makedirs("logs", exist_ok=True)
        log_path = "logs/audit_trail.csv"
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "amt": amt,
            "category": category,
            "gender": gender,
            "hour": hour,
            "probability": round(float(probability), 4),
            "decision": "fraud" if prediction == 1 else "genuine",
        }
        file_exists = os.path.isfile(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        pass  # audit logging should never break the live prediction flow


FEATURE_LABELS = {
    "amt": "Transaction amount",
    "category_encoded": "Merchant category",
    "gender_encoded": "Gender",
    "hour": "Hour of day",
    "city_pop": "City population",
    "lat": "Customer latitude",
    "long": "Customer longitude",
    "merch_lat": "Merchant latitude",
    "merch_long": "Merchant longitude",
}
FEATURES = list(FEATURE_LABELS.keys())

df = load_data()
metrics = load_metrics()
fraud_count = int(df['is_fraud'].sum())
fraud_rate = (fraud_count / len(df)) * 100


def style_fig(fig, height=380, showlegend=False):
    fig.update_layout(
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif", size=13),
        title=dict(text="", font=dict(family="Space Grotesk, sans-serif", size=15, color=TEXT_COLOR)),
        margin=dict(l=10, r=10, t=20, b=10), height=height, showlegend=showlegend,
        hoverlabel=dict(bgcolor="#1B2340", font_color=TEXT_COLOR, bordercolor=GRID_COLOR),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, showline=False)
    return fig


# =======================================================================
# TOP BAR
# =======================================================================
st.markdown("""
<div class="topbar">
  <div class="topbar-brand"><span class="mark">💠</span> FinSight AI</div>
  <div class="topbar-links">
    <a href="https://www.kaggle.com/datasets/kartik2112/fraud-detection" target="_blank">Dataset ↗</a>
    <a href="#" class="status-pill"><span class="pulse"></span> System Operational</a>
  </div>
</div>
""", unsafe_allow_html=True)

# =======================================================================
# HERO
# =======================================================================
st.markdown(f"""
<div class="hero">
  <div class="hero-eyebrow"><span class="dot"></span> RAZORPAY AI BUILDATHON · TRACK 02: AI RISK MANAGER</div>
  <h1>💰 Fraud detection that shows its work</h1>
  <div class="hero-sub">
    A risk engine for card-not-present transactions — evaluated on a <b>held-out test set</b>
    (never seen during training), and priced by business impact, not just accuracy. Built on the
    <a href="https://www.kaggle.com/datasets/kartik2112/fraud-detection" target="_blank">Kaggle Credit Card
    Transactions Fraud Detection (simulated) dataset</a>.
  </div>
  <div class="badge-row">
    <span class="badge live">● live demo</span>
    <span class="badge">🌲 Random Forest · class-balanced</span>
    <span class="badge">🎯 {metrics['test_set_size']:,} held-out transactions</span>
    <span class="badge">🧪 stratified 80/20 split</span>
  </div>
  <div style="font-size:0.78rem; color:{MUTED}; margin: -8px 0 18px; display:flex; align-items:center; gap:6px;">
    🛡️ Defense-only: this system scores and flags transactions for human/automated review — it never autonomously blocks, reverses, or retaliates against a transaction.
  </div>
  <div class="kpi-strip">
    <div class="kpi"><div class="kpi-label">Precision</div><div class="kpi-value">{metrics['precision']*100:.1f}%</div></div>
    <div class="kpi"><div class="kpi-label">Recall</div><div class="kpi-value">{metrics['recall']*100:.1f}%</div></div>
    <div class="kpi"><div class="kpi-label">F1 Score</div><div class="kpi-value">{metrics['f1_score']*100:.1f}%</div></div>
    <div class="kpi"><div class="kpi-label">ROC-AUC</div><div class="kpi-value">{metrics['roc_auc']:.3f}</div></div>
    <div class="kpi"><div class="kpi-label">Fraud in dataset</div><div class="kpi-value warn">{fraud_rate:.1f}%</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_perf, tab_insights, tab_live, tab_batch = st.tabs(
    ["📊  Overview", "🧪  Model Performance", "🔍  Fraud Insights", "🎯  Live Prediction", "📂  Batch Scoring"]
)

# =======================================================================
# TAB 1 — OVERVIEW
# =======================================================================
with tab_overview:
    col1, col2 = st.columns([1.3, 1])

    with col1:
        with st.container(border=True):
            card_header("📈", "Dataset at a glance", "What the model was trained and tested on")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Transactions", f"{len(df):,}")
            m2.metric("Fraud Cases", f"{fraud_count:,}")
            m3.metric("Fraud Rate (dataset)", f"{fraud_rate:.2f}%")
            st.caption(
                "⚠️ This dataset's fraud rate (~15.8%) is far higher than real-world card "
                "fraud (typically <1%). It's simulated/sampled for pipeline demo purposes, "
                "not a claim about real transaction volumes."
            )
            st.markdown("&nbsp;")
            st.markdown("**What this product does**")
            st.markdown(
                "- Classifies a transaction as genuine or fraudulent using amount, category, "
                "gender, hour of day, and location features.\n"
                "- Reports performance on data the model **never saw during training**.\n"
                "- Breaks down the *cost* of getting it wrong in each direction.\n"
                "- Lets you drag the decision threshold and watch the trade-off move live.\n"
                "- Tests arbitrary transactions with a **true per-transaction SHAP explanation** "
                "(not just a global ranking).\n"
                "- Scores a whole CSV of transactions at once and flags the riskiest ones."
            )

    with col2:
        with st.container(border=True):
            card_header("🍩", "Composition", "Genuine vs fraud split")
            donut = go.Figure(data=[go.Pie(
                labels=["Genuine", "Fraud"],
                values=[len(df) - fraud_count, fraud_count],
                hole=0.62,
                marker=dict(colors=[SAFE, DANGER], line=dict(color=PANEL_BG, width=3)),
                textinfo="percent",
                textfont=dict(color=TEXT_COLOR, size=13, family="JetBrains Mono"),
            )])
            donut.add_annotation(
                text=f"{fraud_rate:.1f}%<br><span style='font-size:11px;color:{MUTED}'>fraud</span>",
                showarrow=False, font=dict(size=22, color=DANGER, family="JetBrains Mono"))
            donut = style_fig(donut, height=272)
            st.plotly_chart(donut, use_container_width=True)

    with st.container(border=True):
        card_header("🛠️", "How It Works", "From raw transactions to a live fraud decision")
        s1, s2, s3, s4 = st.columns(4)
        steps = [
            ("1", "Split honestly", "80/20 stratified train/test split — the test set is never shown to the model during training."),
            ("2", "Train", "A class-balanced Random Forest learns from amount, category, gender, hour, and location features."),
            ("3", "Evaluate", "Precision, recall, F1, and ROC-AUC computed only on the held-out test set — no training-set leakage."),
            ("4", "Price the errors", "False positives and false negatives are converted into estimated dollar cost, not left as raw counts."),
        ]
        for col, (num, title, desc) in zip([s1, s2, s3, s4], steps):
            with col:
                st.markdown(f"""
                <div style="height:100%;">
                  <div style="font-family:'JetBrains Mono',monospace; color:{ACCENT}; font-size:0.78rem; margin-bottom:4px;">STEP {num}</div>
                  <div style="font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:0.95rem; color:{TEXT_COLOR}; margin-bottom:4px;">{title}</div>
                  <div style="font-size:0.82rem; color:{MUTED}; line-height:1.45;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    with st.container(border=True):
        card_header("📋", "Sample Transactions", "First 20 rows of the working dataset")
        st.dataframe(df.head(20), use_container_width=True)

# =======================================================================
# TAB 2 — MODEL PERFORMANCE
# =======================================================================
with tab_perf:
    st.markdown(
        f"<div style='color:{MUTED}; margin: 4px 0 16px;'>Evaluated on "
        f"<b style='color:{TEXT_COLOR}'>{metrics['test_set_size']:,} transactions</b> the model "
        f"never saw during training (trained on {metrics['train_set_size']:,}).</div>",
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{metrics['precision']*100:.1f}%")
    m2.metric("Recall", f"{metrics['recall']*100:.1f}%")
    m3.metric("F1 Score", f"{metrics['f1_score']*100:.1f}%")
    m4.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

    cm = metrics['confusion_matrix']
    left, right = st.columns([1, 1.15])

    with left:
        with st.container(border=True):
            card_header("🧮", "Confusion Matrix", "Predictions vs. reality on the test set")
            z = [[cm['true_negative'], cm['false_positive']],
                 [cm['false_negative'], cm['true_positive']]]
            labels = [["True Negative", "False Positive"], ["False Negative", "True Positive"]]
            cm_fig = go.Figure(data=go.Heatmap(
                z=z, x=["Predicted: Genuine", "Predicted: Fraud"], y=["Actual: Genuine", "Actual: Fraud"],
                colorscale=[[0, PANEL_BG], [0.5, "#7A2436"], [1, DANGER]],
                text=[[f"{labels[i][j]}<br><b>{z[i][j]:,}</b>" for j in range(2)] for i in range(2)],
                texttemplate="%{text}", textfont=dict(color="#F5F7FA", size=13), showscale=False,
                hovertemplate="%{y} → %{x}<br>Count: %{z:,}<extra></extra>", xgap=4, ygap=4,
            ))
            cm_fig.update_yaxes(autorange="reversed")
            cm_fig = style_fig(cm_fig, height=320)
            st.plotly_chart(cm_fig, use_container_width=True)

    with right:
        with st.container(border=True):
            card_header("💸", "False-Positive vs False-Negative Cost", "Where the money actually goes when the model is wrong")
            c1, c2 = st.columns(2)
            with c1:
                st.metric(f"False Positives ({cm['false_positive']} txns)",
                          f"${metrics['false_positive_cost_estimate_usd']:,.2f}")
                st.caption("Genuine transactions wrongly flagged — customer friction, support load.")
            with c2:
                st.metric(f"False Negatives ({cm['false_negative']} txns)",
                          f"${metrics['false_negative_cost_estimate_usd']:,.2f}")
                st.caption("Fraudulent transactions the model missed — direct monetary loss.")
            st.caption(metrics['cost_assumptions'])

        with st.container(border=True):
            card_header("🌲", "Feature Importance", "Which signals drive the model's decisions")
            fi = pd.Series(metrics['feature_importances']).sort_values(ascending=True)
            fi_fig = go.Figure(go.Bar(
                x=fi.values, y=fi.index, orientation='h',
                marker=dict(color=fi.values, colorscale=[[0, "#1B4A44"], [1, ACCENT]]),
                hovertemplate="%{y}: %{x:.1%}<extra></extra>",
                text=[f"{v:.1%}" for v in fi.values], textposition="outside",
                textfont=dict(color=TEXT_COLOR, size=11),
            ))
            fi_fig = style_fig(fi_fig, height=280)
            fi_fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fi_fig, use_container_width=True)

    # -------------------------------------------------------------
    with st.container(border=True):
        card_header("📉", "ROC & Precision-Recall Curves", "Threshold-independent view of separability, computed on the held-out test set")
        _roc_preds = load_test_predictions()
        _y_true_roc = _roc_preds["y_true"].values
        _y_proba_roc = _roc_preds["y_proba"].values

        fpr, tpr, _ = roc_curve(_y_true_roc, _y_proba_roc)
        prec_curve, rec_curve, _ = precision_recall_curve(_y_true_roc, _y_proba_roc)
        ap_score = average_precision_score(_y_true_roc, _y_proba_roc)

        rcol1, rcol2 = st.columns(2)
        with rcol1:
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC",
                                          line=dict(color=ACCENT, width=2.5), fill="tozeroy",
                                          fillcolor="rgba(45,212,191,0.10)"))
            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                          line=dict(color=MUTED, width=1.5, dash="dot")))
            roc_fig.add_annotation(
                x=0.98, y=0.06, xanchor="right", showarrow=False,
                text=f"AUC = {metrics['roc_auc']:.3f}",
                font=dict(family="JetBrains Mono", size=13, color=ACCENT),
            )
            roc_fig.update_xaxes(title="False Positive Rate", range=[0, 1])
            roc_fig.update_yaxes(title="True Positive Rate", range=[0, 1])
            roc_fig = style_fig(roc_fig, height=300, showlegend=False)
            st.markdown(f"<div style='font-size:0.82rem; color:{MUTED}; margin-bottom:4px;'>ROC curve — trade-off between catching fraud and false-alarming on genuine transactions</div>", unsafe_allow_html=True)
            st.plotly_chart(roc_fig, use_container_width=True)
        with rcol2:
            pr_fig = go.Figure()
            pr_fig.add_trace(go.Scatter(x=rec_curve, y=prec_curve, mode="lines", name="PR",
                                         line=dict(color=DANGER, width=2.5), fill="tozeroy",
                                         fillcolor="rgba(251,113,133,0.10)"))
            baseline_rate = _y_true_roc.mean()
            pr_fig.add_trace(go.Scatter(x=[0, 1], y=[baseline_rate, baseline_rate], mode="lines",
                                         name="Baseline (fraud rate)",
                                         line=dict(color=MUTED, width=1.5, dash="dot")))
            pr_fig.add_annotation(
                x=0.02, y=0.06, xanchor="left", showarrow=False,
                text=f"Avg. Precision = {ap_score:.3f}",
                font=dict(family="JetBrains Mono", size=13, color=DANGER),
            )
            pr_fig.update_xaxes(title="Recall", range=[0, 1])
            pr_fig.update_yaxes(title="Precision", range=[0, 1.02])
            pr_fig = style_fig(pr_fig, height=300, showlegend=False)
            st.markdown(f"<div style='font-size:0.82rem; color:{MUTED}; margin-bottom:4px;'>Precision-Recall curve — more informative than ROC when the positive class (fraud) is rare</div>", unsafe_allow_html=True)
            st.plotly_chart(pr_fig, use_container_width=True)

        st.caption(
            "Both curves are threshold-independent — they show how good the model's *ranking* of "
            "transactions by risk is, before any 0.50 cutoff is applied. A model with high ROC-AUC "
            "but a weaker PR curve would still be worth flagging, since PR is the harder test on "
            "an imbalanced problem like fraud."
        )

    # -------------------------------------------------------------
    with st.container(border=True):
        card_header("🎚️", "Tune the Decision Threshold", "See the trade-off move in real time as you drag")
        st.markdown(
            "The model outputs a *fraud probability*, not a yes/no. Right now, anything above "
            "**0.50** gets flagged as fraud. Move the slider to see how that choice trades off "
            "false positives against false negatives — this is the real decision a risk team has to make."
        )

        test_preds = load_test_predictions()
        threshold = st.slider("Flag as fraud if probability ≥", 0.05, 0.95, 0.50, 0.05, key="threshold_slider")

        y_true_t = test_preds["y_true"].values
        y_proba_t = test_preds["y_proba"].values
        y_pred_t = (y_proba_t >= threshold).astype(int)

        tn_t = int(((y_true_t == 0) & (y_pred_t == 0)).sum())
        fp_t = int(((y_true_t == 0) & (y_pred_t == 1)).sum())
        fn_t = int(((y_true_t == 1) & (y_pred_t == 0)).sum())
        tp_t = int(((y_true_t == 1) & (y_pred_t == 1)).sum())

        precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
        recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
        f1_t = 2 * precision_t * recall_t / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0.0

        fp_amt_t = test_preds.loc[(test_preds.y_true == 0) & (y_pred_t == 1), "amt"].sum()
        fn_amt_t = test_preds.loc[(test_preds.y_true == 1) & (y_pred_t == 0), "amt"].sum()

        tcol1, tcol2, tcol3 = st.columns(3)

        def fmt_delta(diff_pp):
            if abs(diff_pp) < 0.05:
                return None
            return f"{diff_pp:+.1f} pp vs default"

        tcol1.metric("Precision", f"{precision_t*100:.1f}%", delta=fmt_delta((precision_t - metrics['precision'])*100))
        tcol2.metric("Recall", f"{recall_t*100:.1f}%", delta=fmt_delta((recall_t - metrics['recall'])*100))
        tcol3.metric("F1 Score", f"{f1_t*100:.1f}%", delta=fmt_delta((f1_t - metrics['f1_score'])*100))

        thresh_grid = np.arange(0.05, 0.96, 0.02)
        precisions, recalls, f1s = [], [], []
        for t in thresh_grid:
            pred = (y_proba_t >= t).astype(int)
            tp = int(((y_true_t == 1) & (pred == 1)).sum())
            fp = int(((y_true_t == 0) & (pred == 1)).sum())
            fn = int(((y_true_t == 1) & (pred == 0)).sum())
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precisions.append(p); recalls.append(r); f1s.append(f)

        curve_fig = go.Figure()
        curve_fig.add_trace(go.Scatter(x=thresh_grid, y=precisions, name="Precision",
                                        line=dict(color=ACCENT, width=2.5), mode="lines"))
        curve_fig.add_trace(go.Scatter(x=thresh_grid, y=recalls, name="Recall",
                                        line=dict(color=DANGER, width=2.5), mode="lines"))
        curve_fig.add_trace(go.Scatter(x=thresh_grid, y=f1s, name="F1",
                                        line=dict(color=MUTED, width=2, dash="dot"), mode="lines"))
        curve_fig.add_vline(x=threshold, line_width=1.5, line_dash="dash", line_color=ACCENT_SOFT,
                             annotation_text="current", annotation_font_color=ACCENT_SOFT)
        curve_fig.update_yaxes(tickformat=".0%", title="Score")
        curve_fig.update_xaxes(title="Decision threshold")
        curve_fig = style_fig(curve_fig, height=320, showlegend=True)
        curve_fig.update_layout(legend=dict(orientation="h", y=1.14, x=0))
        st.plotly_chart(curve_fig, use_container_width=True)

        tcost1, tcost2 = st.columns(2)
        with tcost1:
            st.metric(f"False Positives ({fp_t} txns)", f"${fp_amt_t*0.02:,.2f} est. friction cost")
            st.caption("Genuine transactions blocked at this threshold.")
        with tcost2:
            st.metric(f"False Negatives ({fn_t} txns)", f"${fn_amt_t:,.2f} est. fraud loss")
            st.caption("Fraud missed at this threshold.")

        if threshold < 0.5:
            st.info("🔽 **Lower threshold** — catches more fraud (higher recall) but blocks more genuine "
                    "customers too. Good when missed fraud is very costly.")
        elif threshold > 0.5:
            st.info("🔼 **Higher threshold** — fewer genuine customers blocked (higher precision) but more "
                    "fraud slips through. Good when customer friction is the bigger business risk.")
        else:
            st.info("This is the model's default threshold (0.50).")

    # -------------------------------------------------------------
    if "slice_checks" in metrics:
        with st.container(border=True):
            card_header(
                "🧬", "Slice Checks",
                "Precision/recall on harder subsets — not just one aggregate number"
            )
            st.caption(
                "A single aggregate score can hide weak spots. These are the same "
                "precision/recall metrics, recomputed only on two slices of the "
                "held-out test set that are more likely to stress the model."
            )
            sc = metrics["slice_checks"]
            slice_cols = st.columns(len(sc))
            for col, (key, s) in zip(slice_cols, sc.items()):
                with col:
                    label = key.replace("_", " ").title()
                    if s.get("n", 0) == 0 or s.get("precision") is None:
                        st.metric(label, "n/a")
                        st.caption("No transactions in this slice.")
                    else:
                        st.metric(
                            label, f"P {s['precision']*100:.1f}% / R {s['recall']*100:.1f}%",
                            help=s.get("description", ""),
                        )
                        st.caption(f"{s['n']:,} transactions · {s.get('description', '')}")

    with st.expander("Known limitations"):
        st.markdown("""
- Trained on a simulated dataset — fraud patterns and rates won't exactly match a real payments environment.
- Location fields (`lat`/`long`) are static per-user rather than per-transaction GPS, which may inflate their apparent importance.
- No time-series / velocity features yet (e.g. "transactions in the last 10 minutes") — a common real fraud signal that this version doesn't capture.
- Model is retrained as a batch job (`train_model.py`), not online/incrementally.
""")

# =======================================================================
# TAB 3 — FRAUD INSIGHTS
# =======================================================================
with tab_insights:
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            card_header("💵", "Amount: Genuine vs Fraud", "Distribution comparison")
            box_df = df.copy()
            box_df['Label'] = box_df['is_fraud'].map({0: "Genuine", 1: "Fraud"})
            box_fig = px.box(box_df, x='Label', y='amt', color='Label',
                              color_discrete_map={"Genuine": SAFE, "Fraud": DANGER}, points=False)
            box_fig.update_traces(marker_line_color=GRID_COLOR)
            box_fig.update_yaxes(title="Amount ($)")
            box_fig.update_xaxes(title="")
            box_fig = style_fig(box_fig, height=360)
            st.plotly_chart(box_fig, use_container_width=True)

    with col2:
        with st.container(border=True):
            card_header("🏷️", "Fraud Rate by Category", "Which merchant categories carry the most risk")
            fraud_by_category = (df.groupby('category')['is_fraud'].mean().sort_values(ascending=False) * 100)
            cat_fig = go.Figure(go.Bar(
                x=fraud_by_category.index, y=fraud_by_category.values,
                marker=dict(color=fraud_by_category.values, colorscale=[[0, "#7A2436"], [1, DANGER]]),
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            ))
            cat_fig.update_yaxes(title="Fraud Rate (%)")
            cat_fig.update_xaxes(title="", tickangle=-40)
            cat_fig = style_fig(cat_fig, height=360)
            st.plotly_chart(cat_fig, use_container_width=True)

    with st.container(border=True):
        card_header("🕐", "Fraud Rate by Hour of Day", "When risk spikes across a 24-hour cycle")
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['hour'] = df['trans_date_trans_time'].dt.hour
        fraud_by_hour = df.groupby('hour')['is_fraud'].mean() * 100
        hour_fig = go.Figure(go.Scatter(
            x=fraud_by_hour.index, y=fraud_by_hour.values, mode="lines+markers",
            line=dict(color=ACCENT, width=2.5), fill="tozeroy", fillcolor="rgba(45,212,191,0.12)",
            marker=dict(color=DANGER, size=7, line=dict(color=PANEL_BG, width=1)),
            hovertemplate="Hour %{x}:00 — %{y:.1f}%<extra></extra>",
        ))
        hour_fig.update_xaxes(title="Hour (24-hour format)", dtick=1)
        hour_fig.update_yaxes(title="Fraud Rate (%)")
        hour_fig = style_fig(hour_fig, height=320)
        st.plotly_chart(hour_fig, use_container_width=True)

# =======================================================================
# TAB 4 — LIVE PREDICTION
# =======================================================================
with tab_live:
    model = joblib.load('models/fraud_model.pkl')
    le_category = joblib.load('models/le_category.pkl')
    le_gender = joblib.load('models/le_gender.pkl')

    # ---------------------------------------------------------------
    # One-click presets — a judge/reviewer shouldn't have to hand-fill
    # 7 fields to see the model work. Drawn from real dataset stats:
    # shopping_net has the dataset's highest fraud rate (~36%), and
    # fraud amounts skew much higher than genuine ones.
    # ---------------------------------------------------------------
    PRESETS = {
        "genuine": dict(amt=42.50, category="grocery_pos", gender="F", hour=14, city_pop=85000, lat=40.71, long=-74.01),
        "suspicious": dict(amt=980.00, category="shopping_net", gender="M", hour=3, city_pop=1200, lat=36.17, long=-115.14),
    }

    def apply_preset(name):
        p = PRESETS[name]
        st.session_state["amt_input"] = p["amt"]
        st.session_state["category_input"] = p["category"]
        st.session_state["gender_input"] = p["gender"]
        st.session_state["hour_input"] = p["hour"]
        st.session_state["citypop_input"] = p["city_pop"]
        st.session_state["lat_input"] = p["lat"]
        st.session_state["long_input"] = p["long"]
        st.session_state["auto_check"] = True

    with st.container(border=True):
        card_header("🧾", "Test a Transaction", "Enter details to check the model's live fraud probability")

        st.caption("Not sure what to try? Load an example:")
        pcol1, pcol2, pcol3 = st.columns([1, 1, 2])
        with pcol1:
            if st.button("🟢 Genuine example", use_container_width=True):
                apply_preset("genuine")
                st.rerun()
        with pcol2:
            if st.button("🔴 Suspicious example", use_container_width=True):
                apply_preset("suspicious")
                st.rerun()

        col1, col2, col3 = st.columns(3)
        with col1:
            amt = st.number_input("Transaction Amount ($)", min_value=0.0, step=10.0, key="amt_input", value=st.session_state.get("amt_input", 100.0))
            category = st.selectbox("Category", le_category.classes_, key="category_input",
                                     index=list(le_category.classes_).index(st.session_state.get("category_input", le_category.classes_[0])))
        with col2:
            gender = st.selectbox("Gender", le_gender.classes_, key="gender_input",
                                   index=list(le_gender.classes_).index(st.session_state.get("gender_input", le_gender.classes_[0])))
            hour = st.slider("Hour of Transaction (24-hr)", 0, 23, key="hour_input", value=st.session_state.get("hour_input", 12))
        with col3:
            city_pop = st.number_input("City Population", min_value=0, step=1000, key="citypop_input", value=st.session_state.get("citypop_input", 50000))
            lat = st.number_input("Latitude", format="%.4f", key="lat_input", value=st.session_state.get("lat_input", 40.0))
        long = st.number_input("Longitude", format="%.4f", key="long_input", value=st.session_state.get("long_input", -75.0))
        check = st.button("🔍 Check for Fraud", type="primary") or st.session_state.pop("auto_check", False)

    # Compute and STORE the result in session_state on check. This is the key
    # fix: without this, clicking the nested "Generate explanation" button
    # below triggers a rerun where `check` is False again, which used to wipe
    # out this entire results section. Storing in session_state makes the
    # result persist across that rerun.
    if check:
        category_encoded = le_category.transform([category])[0]
        gender_encoded = le_gender.transform([gender])[0]

        input_data = pd.DataFrame({
            'amt': [amt], 'category_encoded': [category_encoded], 'gender_encoded': [gender_encoded],
            'hour': [hour], 'city_pop': [city_pop], 'lat': [lat], 'long': [long],
            'merch_lat': [lat], 'merch_long': [long]
        })

        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])

        log_prediction(amt, category, gender, hour, probability, prediction, source="live")

        with st.spinner("Analyzing transaction..."):
            explainer = get_shap_explainer(model)
            shap_raw = explainer.shap_values(input_data)
            if isinstance(shap_raw, list):
                sv = np.array(shap_raw[1])[0]
            elif np.ndim(shap_raw) == 3:
                sv = shap_raw[0, :, 1]
            else:
                sv = shap_raw[0]

            shap_df = pd.DataFrame({
                "feature": [FEATURE_LABELS.get(f, f) for f in FEATURES],
                "value": [input_data[f].iloc[0] for f in FEATURES],
                "shap": sv,
            }).sort_values("shap", key=lambda s: s.abs(), ascending=True)

        st.session_state["live_result"] = {
            "prediction": prediction,
            "probability": probability,
            "shap_df": shap_df,
        }
        # Clear any previous LLM explanation — it belongs to the old transaction.
        st.session_state.pop("live_explanation", None)

    if "live_result" not in st.session_state:
        st.info(
            "👆 Load an example above, or fill in the form and hit **Check for Fraud** "
            "to see a live prediction with a full SHAP breakdown."
        )

    if "live_result" in st.session_state:
        result = st.session_state["live_result"]
        prediction = result["prediction"]
        probability = result["probability"]
        shap_df = result["shap_df"]

        res_col, gauge_col = st.columns([1.4, 1])
        with res_col:
            with st.container(border=True):
                card_header("🎯" if prediction == 0 else "🚨", "Prediction Result")
                if prediction == 1:
                    st.error(f"⚠️ FRAUD ALERT! This transaction has a {probability*100:.1f}% probability of being fraudulent.")
                else:
                    st.success(f"✅ This transaction looks genuine. Fraud probability: {probability*100:.1f}%")

                st.markdown("**Why the model leans this way** — SHAP attribution for *this exact transaction*:")

                shap_fig = go.Figure(go.Bar(
                    x=shap_df["shap"], y=shap_df["feature"], orientation="h",
                    marker=dict(color=[DANGER if v > 0 else SAFE for v in shap_df["shap"]]),
                    hovertemplate="%{y}<br>SHAP: %{x:+.4f}<extra></extra>",
                    text=[f"{v:+.3f}" for v in shap_df["shap"]], textposition="outside",
                    textfont=dict(color=TEXT_COLOR, size=11),
                ))
                shap_fig.add_vline(x=0, line_width=1, line_color=GRID_COLOR)
                shap_fig.update_xaxes(title="← pushes toward genuine   |   pushes toward fraud →")
                shap_fig = style_fig(shap_fig, height=280)
                st.plotly_chart(shap_fig, use_container_width=True)

                top_driver = shap_df.iloc[-1]
                direction = "toward fraud" if top_driver["shap"] > 0 else "toward genuine"
                st.caption(
                    f"Biggest single driver for **this** transaction: **{top_driver['feature']}** "
                    f"(value {top_driver['value']:.2f}), pushing the prediction {direction}. "
                    "Unlike a global feature-importance ranking, these SHAP values are computed "
                    "specifically for this transaction's inputs — this is the true per-prediction "
                    "attribution the earlier version of this dashboard was missing."
                )

                with st.expander("🤖 Ask AI to explain this in plain English"):
                    if st.button("Generate explanation", key="llm_explain_btn"):
                        with st.spinner("Generating explanation..."):
                            explanation, err = explain_with_llm(shap_df, probability, prediction)
                        st.session_state["live_explanation"] = (explanation, err)

                    if "live_explanation" in st.session_state:
                        explanation, err = st.session_state["live_explanation"]
                        if explanation:
                            st.markdown(explanation)
                            st.caption(
                                "Generated by an LLM to narrate the SHAP values above for a human "
                                "reader — it does not influence the model's decision."
                            )
                        else:
                            st.info(err)

        with gauge_col:
            with st.container(border=True):
                card_header("📟", "Fraud Probability")
                gauge_color = DANGER if probability >= 0.5 else SAFE
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={'suffix': "%", 'font': {'color': gauge_color, 'family': 'JetBrains Mono', 'size': 34}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': MUTED, 'tickfont': {'color': MUTED}},
                        'bar': {'color': gauge_color},
                        'bgcolor': PANEL_BG, 'borderwidth': 1, 'bordercolor': GRID_COLOR,
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(52,211,153,0.15)"},
                            {'range': [50, 100], 'color': "rgba(251,113,133,0.15)"},
                        ],
                        'threshold': {'line': {'color': ACCENT_SOFT, 'width': 3}, 'thickness': 0.8, 'value': 50},
                    },
                ))
                gauge = style_fig(gauge, height=260)
                st.plotly_chart(gauge, use_container_width=True)

# =======================================================================
# TAB 5 — BATCH SCORING
# =======================================================================
with tab_batch:
    model_b = joblib.load('models/fraud_model.pkl')
    le_category_b = joblib.load('models/le_category.pkl')
    le_gender_b = joblib.load('models/le_gender.pkl')

    with st.container(border=True):
        card_header("📂", "Score a Batch of Transactions", "Upload a CSV and flag every risky transaction in one pass")
        st.markdown(
            "Upload a CSV with the same columns as the training data "
            "(`amt`, `category`, `gender`, `city_pop`, `lat`, `long`, and either "
            "`hour` or `trans_date_trans_time`). `merch_lat` / `merch_long` are optional — "
            "if missing, customer lat/long are reused."
        )

        sample_template = df.head(5)[[
            "trans_date_trans_time", "category", "gender", "amt", "city_pop", "lat", "long",
            "merch_lat", "merch_long"
        ]] if set(["trans_date_trans_time", "category", "gender", "amt", "city_pop", "lat", "long",
                    "merch_lat", "merch_long"]).issubset(df.columns) else None

        tcol1, tcol2 = st.columns([3, 1])
        with tcol1:
            uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
        with tcol2:
            if sample_template is not None:
                st.download_button(
                    "⬇️ Sample template", data=sample_template.to_csv(index=False),
                    file_name="batch_template.csv", mime="text/csv", use_container_width=True,
                )

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't read that CSV: {e}")
            batch_df = None

        if batch_df is not None:
            missing_required = [c for c in ["amt", "category", "gender"] if c not in batch_df.columns]
            if missing_required:
                st.error(f"Missing required column(s): {', '.join(missing_required)}")
            else:
                work = batch_df.copy()

                if "hour" not in work.columns:
                    if "trans_date_trans_time" in work.columns:
                        work["hour"] = pd.to_datetime(work["trans_date_trans_time"], errors="coerce").dt.hour
                    else:
                        work["hour"] = 12  # neutral default, flagged below

                for col, default in [("city_pop", 50000), ("lat", 40.0), ("long", -75.0)]:
                    if col not in work.columns:
                        work[col] = default
                if "merch_lat" not in work.columns:
                    work["merch_lat"] = work["lat"]
                if "merch_long" not in work.columns:
                    work["merch_long"] = work["long"]

                # Unseen categories/genders would break the label encoder — flag and
                # drop those rows rather than crashing the whole batch.
                known_cats = set(le_category_b.classes_)
                known_genders = set(le_gender_b.classes_)
                bad_rows = ~work["category"].isin(known_cats) | ~work["gender"].isin(known_genders)
                n_bad = int(bad_rows.sum())
                if n_bad:
                    st.warning(
                        f"⚠️ Skipped {n_bad} row(s) with a category/gender value the model wasn't "
                        f"trained on. Known categories: {', '.join(sorted(known_cats))}."
                    )
                clean = work.loc[~bad_rows].copy()

                if len(clean) == 0:
                    st.error("No valid rows left to score after filtering.")
                else:
                    clean["category_encoded"] = le_category_b.transform(clean["category"])
                    clean["gender_encoded"] = le_gender_b.transform(clean["gender"])
                    clean["hour"] = clean["hour"].fillna(12).astype(int)

                    X_batch = clean[FEATURES]
                    clean["fraud_probability"] = model_b.predict_proba(X_batch)[:, 1]
                    clean["predicted_fraud"] = (clean["fraud_probability"] >= 0.5).astype(int)

                    n_flagged = int(clean["predicted_fraud"].sum())
                    est_value_flagged = clean.loc[clean["predicted_fraud"] == 1, "amt"].sum()

                    log_prediction(
                        amt=float(clean["amt"].sum()), category="(batch)", gender="(batch)",
                        hour=None, probability=float(clean["fraud_probability"].mean()),
                        prediction=1 if n_flagged > 0 else 0, source=f"batch:{len(clean)}_rows",
                    )

                    with st.container(border=True):
                        card_header("📊", "Batch Results", f"{len(clean):,} transactions scored")
                        s1, s2, s3 = st.columns(3)
                        s1.metric("Transactions scored", f"{len(clean):,}")
                        s2.metric("Flagged as fraud", f"{n_flagged:,}", delta=f"{n_flagged/len(clean)*100:.1f}% of batch")
                        s3.metric("Value flagged", f"${est_value_flagged:,.2f}")

                        display_cols = ["amt", "category", "gender", "hour", "fraud_probability", "predicted_fraud"]
                        display_cols = [c for c in display_cols if c in clean.columns]
                        result_view = clean[display_cols].sort_values("fraud_probability", ascending=False)
                        st.dataframe(
                            result_view.style.format({"fraud_probability": "{:.1%}", "amt": "${:.2f}"}),
                            use_container_width=True, height=360,
                        )

                        st.download_button(
                            "⬇️ Download scored CSV",
                            data=clean.drop(columns=["category_encoded", "gender_encoded"], errors="ignore").to_csv(index=False),
                            file_name="scored_transactions.csv", mime="text/csv",
                        )

                    with st.container(border=True):
                        card_header("📈", "Risk Distribution", "Fraud probability spread across this batch")
                        hist_fig = go.Figure(go.Histogram(
                            x=clean["fraud_probability"], nbinsx=30,
                            marker=dict(color=ACCENT, line=dict(color=PANEL_BG, width=1)),
                        ))
                        hist_fig.add_vline(x=0.5, line_width=1.5, line_dash="dash", line_color=DANGER,
                                            annotation_text="0.50 cutoff", annotation_font_color=DANGER)
                        hist_fig.update_xaxes(title="Predicted fraud probability", tickformat=".0%")
                        hist_fig.update_yaxes(title="Number of transactions")
                        hist_fig = style_fig(hist_fig, height=280)
                        st.plotly_chart(hist_fig, use_container_width=True)
    else:
        st.caption(
            "No file uploaded yet — grab the sample template above, or export a slice of your "
            "own transaction log with the same columns as `data/transactions_sample.csv`."
        )

# =======================================================================
# FOOTER
# =======================================================================
st.markdown("""
<div class="app-footer">
  <div>Built for the Razorpay AI Buildathon — Track 02: AI Risk Manager</div>
  <div class="badges">
    <span>Streamlit</span><span>scikit-learn</span><span>Plotly</span><span>Rendered on Render</span>
  </div>
</div>
""", unsafe_allow_html=True)
