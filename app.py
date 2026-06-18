import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import io, warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopIntent · ML Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ────────────────────────────────────────────────────────────
# Signature palette: deep navy base, electric teal accent, warm amber highlight
C = {
    "navy":    "#0B0F1A",
    "navy2":   "#111827",
    "navy3":   "#1C2333",
    "navy4":   "#232B3E",
    "teal":    "#00D4AA",
    "teal2":   "#00A87E",
    "teal3":   "#00FFD0",
    "amber":   "#F59E0B",
    "amber2":  "#FCD34D",
    "rose":    "#F43F5E",
    "violet":  "#7C3AED",
    "slate":   "#94A3B8",
    "border":  "#1E293B",
    "white":   "#F1F5F9",
    "white2":  "#CBD5E1",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');

/* ── Global ── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, .stApp {{ background: {C['navy']} !important; font-family: 'DM Sans', sans-serif; color: {C['white']}; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {C['navy2']} !important;
    border-right: 1px solid {C['border']};
}}
[data-testid="stSidebar"] .stMarkdown p {{ color: {C['slate']}; font-size: 13px; }}

/* ── Main content padding ── */
.block-container {{ padding: 2rem 2.5rem 3rem; max-width: 1400px; }}

/* ── Metric cards ── */
[data-testid="stMetric"] {{
    background: {C['navy3']};
    border: 1px solid {C['border']};
    border-radius: 14px;
    padding: 20px 22px !important;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}}
[data-testid="stMetric"]:hover {{ border-color: {C['teal2']}; }}
[data-testid="stMetricLabel"] {{ color: {C['slate']} !important; font-size: 12px !important; font-weight: 500 !important; letter-spacing: 0.06em; text-transform: uppercase; }}
[data-testid="stMetricValue"] {{ color: {C['white']} !important; font-size: 28px !important; font-weight: 700 !important; font-family: 'DM Mono', monospace !important; }}
[data-testid="stMetricDelta"] {{ font-size: 12px !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {C['navy2']};
    border-radius: 12px;
    padding: 5px 6px;
    gap: 4px;
    border: 1px solid {C['border']};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    color: {C['slate']};
    font-weight: 500;
    font-size: 14px;
    padding: 8px 18px;
    transition: all 0.18s;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {C['teal2']}, {C['teal']}) !important;
    color: {C['navy']} !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 12px rgba(0,212,170,0.35);
}}

/* ── Buttons ── */
.stButton > button {{
    background: linear-gradient(135deg, {C['teal2']}, {C['teal']});
    color: {C['navy']};
    border: none;
    border-radius: 10px;
    padding: 12px 28px;
    font-weight: 700;
    font-size: 15px;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.01em;
    width: 100%;
    cursor: pointer;
    transition: all 0.18s;
    box-shadow: 0 4px 18px rgba(0,212,170,0.3);
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(0,212,170,0.45);
}}
.stDownloadButton > button {{
    background: linear-gradient(135deg, {C['amber2']}, {C['amber']});
    color: {C['navy']};
    border: none;
    border-radius: 10px;
    font-weight: 700;
    width: 100%;
    box-shadow: 0 4px 18px rgba(245,158,11,0.3);
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background: {C['navy3']};
    border: 2px dashed {C['border']};
    border-radius: 14px;
    transition: border-color 0.2s;
}}
[data-testid="stFileUploader"]:hover {{ border-color: {C['teal2']}; }}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {{
    background: {C['navy3']} !important;
    border-color: {C['border']} !important;
    border-radius: 10px !important;
    color: {C['white']} !important;
}}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
    background: {C['teal']};
    border-color: {C['teal']};
}}
.stSlider [data-baseweb="slider"] div[style*="background"] {{
    background: {C['teal']} !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] iframe {{ border-radius: 12px; }}

/* ── Dividers ── */
hr {{ border-color: {C['border']} !important; margin: 1.5rem 0 !important; }}

/* ── Custom cards ── */
.kpi-strip {{
    display: flex; gap: 12px; margin: 18px 0;
}}
.hero-banner {{
    background: linear-gradient(135deg, {C['navy3']} 0%, {C['navy4']} 100%);
    border: 1px solid {C['border']};
    border-left: 4px solid {C['teal']};
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 24px;
}}
.section-eyebrow {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {C['teal']};
    margin-bottom: 6px;
}}
.section-title {{
    font-size: 22px;
    font-weight: 700;
    color: {C['white']};
    margin-bottom: 4px;
}}
.section-sub {{
    font-size: 14px;
    color: {C['slate']};
}}
.stat-pill {{
    display: inline-block;
    background: rgba(0,212,170,0.12);
    border: 1px solid rgba(0,212,170,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 600;
    color: {C['teal']};
    margin-right: 6px;
}}
.warn-pill {{
    background: rgba(245,158,11,0.12);
    border-color: rgba(245,158,11,0.3);
    color: {C['amber']};
}}
.info-box {{
    background: {C['navy3']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
}}
.pred-card {{
    background: {C['navy3']};
    border-radius: 12px;
    padding: 18px 22px;
    border: 1px solid {C['border']};
    margin-bottom: 10px;
}}
.tag-yes {{
    background: rgba(0,212,170,0.15);
    border: 1px solid rgba(0,212,170,0.4);
    color: {C['teal3']};
    border-radius: 6px; padding: 2px 10px;
    font-size: 12px; font-weight: 700;
    font-family: 'DM Mono', monospace;
}}
.tag-no {{
    background: rgba(244,63,94,0.15);
    border: 1px solid rgba(244,63,94,0.4);
    color: #fb7185;
    border-radius: 6px; padding: 2px 10px;
    font-size: 12px; font-weight: 700;
    font-family: 'DM Mono', monospace;
}}

/* scrollbar */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {C['navy2']}; }}
::-webkit-scrollbar-thumb {{ background: {C['border']}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Administrative","Administrative_Duration","Informational",
    "Informational_Duration","ProductRelated","ProductRelated_Duration",
    "BounceRates","ExitRates","PageValues","SpecialDay","Month",
    "OperatingSystems","Browser","Region","TrafficType","VisitorType","Weekend",
]
OUTLIER_COLS = [
    "Administrative_Duration","Informational_Duration",
    "ProductRelated_Duration","BounceRates","ExitRates","PageValues",
]
MODELS = {
    "🌲 Random Forest":       RandomForestClassifier(random_state=42),
    "🧭 Decision Tree":       DecisionTreeClassifier(random_state=42),
    "📍 K-Nearest Neighbors": KNeighborsClassifier(),
    "🔔 Naive Bayes":         GaussianNB(),
}
MODEL_COLORS = {
    "🌲 Random Forest":       C["teal"],
    "🧭 Decision Tree":       C["amber"],
    "📍 K-Nearest Neighbors": "#818CF8",
    "🔔 Naive Bayes":         C["rose"],
}

# ── Chart style ───────────────────────────────────────────────────────────────
def chart_style():
    plt.rcParams.update({
        "figure.facecolor": C["navy2"],
        "axes.facecolor":   C["navy3"],
        "axes.edgecolor":   C["border"],
        "axes.labelcolor":  C["slate"],
        "xtick.color":      C["slate"],
        "ytick.color":      C["slate"],
        "text.color":       C["white"],
        "grid.color":       C["border"],
        "grid.linestyle":   "--",
        "grid.alpha":       0.4,
        "font.family":      "DejaVu Sans",
    })

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(df, fit_scaler=True, scaler=None):
    d = df.copy()
    le = LabelEncoder()
    d["Month"]       = le.fit_transform(d["Month"].astype(str))
    d["VisitorType"] = le.fit_transform(d["VisitorType"].astype(str))
    d["Weekend"]     = d["Weekend"].astype(int)
    for col in OUTLIER_COLS:
        if col in d.columns:
            Q1, Q3 = d[col].quantile(.25), d[col].quantile(.75)
            IQR = Q3 - Q1
            d[col] = d[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    num = [c for c in d.select_dtypes(include=np.number).columns if c != "Revenue"]
    if fit_scaler:
        scaler = StandardScaler()
        d[num] = scaler.fit_transform(d[num])
        return d, scaler
    else:
        d[num] = scaler.transform(d[num])
        return d, scaler

def train_model(X_tr, X_te, y_tr, y_te, name):
    m = MODELS[name].__class__(**MODELS[name].get_params())
    m.fit(X_tr, y_tr)
    yp  = m.predict(X_te)
    ypr = m.predict_proba(X_te)[:,1] if hasattr(m,"predict_proba") else None
    metrics = {
        "Accuracy":  accuracy_score(y_te, yp),
        "Precision": precision_score(y_te, yp, zero_division=0),
        "Recall":    recall_score(y_te, yp, zero_division=0),
        "F1":        f1_score(y_te, yp, zero_division=0),
        "ROC AUC":   roc_auc_score(y_te, ypr) if ypr is not None else None,
    }
    return m, metrics, confusion_matrix(y_te, yp), yp, ypr

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:20px 4px 10px">
      <div style="font-size:24px;font-weight:800;letter-spacing:-0.5px;color:{C['white']}">🛍️ ShopIntent</div>
      <div style="font-size:12px;color:{C['slate']};margin-top:4px;font-weight:500">Purchase Intent ML Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown(f"<div style='font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:{C['teal']};margin-bottom:12px'>Configuration</div>", unsafe_allow_html=True)

    model_key = st.selectbox("Algorithm", list(MODELS.keys()), index=0)
    test_pct  = st.slider("Test Split %", 10, 40, 20, 5)
    st.divider()

    st.markdown(f"<div style='font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:{C['teal']};margin-bottom:10px'>Required Columns</div>", unsafe_allow_html=True)
    for col in FEATURE_COLS:
        st.markdown(f"<span style='font-family:DM Mono,monospace;font-size:11px;color:{C['slate']};'>· {col}</span>", unsafe_allow_html=True)
    st.divider()
    st.markdown(f"<div style='font-size:11px;color:{C['slate']};text-align:center;'>Online Shoppers Intention<br>Binary Classification</div>", unsafe_allow_html=True)

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
  <div class="section-eyebrow">Purchase Intent Intelligence</div>
  <div style="font-size:30px;font-weight:800;letter-spacing:-0.5px;color:{C['white']};line-height:1.2;">
    Online Shoppers Prediction Studio
  </div>
  <div style="font-size:15px;color:{C['slate']};margin-top:8px;max-width:600px;">
    Upload the shoppers dataset, train & compare 4 ML models, and export predictions — 
    all with real-time confidence scores and interactive visualisations.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop your CSV here — or click to browse",
    type=["csv"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown(f"""
    <div class="info-box" style="text-align:center;padding:40px;">
      <div style="font-size:40px;margin-bottom:12px;">📂</div>
      <div style="font-size:16px;font-weight:600;color:{C['white']};">Upload the Online Shoppers CSV to begin</div>
      <div style="font-size:13px;color:{C['slate']};margin-top:6px;">Expects the standard <code>online_shoppers_intention.csv</code> schema</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

@st.cache_data
def load(f): return pd.read_csv(f)

df_raw = load(uploaded)
missing = [c for c in FEATURE_COLS if c not in df_raw.columns]
if missing:
    st.error(f"Missing columns: {', '.join(missing)}")
    st.stop()

has_target = "Revenue" in df_raw.columns

# Pill summary
rev_rate = ""
if has_target:
    r = df_raw["Revenue"].astype(int) if df_raw["Revenue"].dtype == bool else df_raw["Revenue"]
    rev_rate = f'<span class="stat-pill warn-pill">{r.mean()*100:.1f}% purchase rate</span>'

st.markdown(f"""
<div style="margin:10px 0 22px;display:flex;align-items:center;flex-wrap:wrap;gap:6px;">
  <span class="stat-pill">{len(df_raw):,} rows</span>
  <span class="stat-pill">{df_raw.shape[1]} columns</span>
  <span class="stat-pill">{df_raw.isnull().sum().sum()} missing</span>
  {rev_rate}
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
T1, T2, T3, T4 = st.tabs(["  📊 Data Explorer  ", "  🤖 Model Lab  ", "  🔮 Predict  ", "  📈 Deep Charts  "])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with T1:
    chart_style()
    st.markdown(f"<div class='section-eyebrow' style='margin-top:8px'>Dataset</div><div class='section-title'>Data Explorer</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # KPI row
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Records",  f"{len(df_raw):,}")
    c2.metric("Features", f"{df_raw.shape[1] - (1 if has_target else 0)}")
    c3.metric("Null Cells", f"{df_raw.isnull().sum().sum()}")
    if has_target:
        r = df_raw["Revenue"].astype(int) if df_raw["Revenue"].dtype == bool else df_raw["Revenue"]
        c4.metric("Purchase Rate", f"{r.mean()*100:.1f}%", f"{r.sum():,} buyers")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C['white']};margin-bottom:8px;'>📋 Sample Records</div>", unsafe_allow_html=True)
    st.dataframe(df_raw.head(15), use_container_width=True, height=300)

    col_left, col_right = st.columns([1,1])

    with col_left:
        st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C['white']};margin:16px 0 8px;'>📐 Schema</div>", unsafe_allow_html=True)
        schema = pd.DataFrame({
            "Column": df_raw.dtypes.index,
            "Type":   df_raw.dtypes.astype(str).values,
            "Non-Null": df_raw.notnull().sum().values,
        })
        st.dataframe(schema, use_container_width=True, height=320)

    with col_right:
        st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C['white']};margin:16px 0 8px;'>📊 Statistics</div>", unsafe_allow_html=True)
        st.dataframe(df_raw.describe().round(3), use_container_width=True, height=320)

    # Target dist
    if has_target:
        st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C['white']};margin:22px 0 12px;'>🎯 Target Variable — Revenue</div>", unsafe_allow_html=True)
        r    = df_raw["Revenue"].astype(int) if df_raw["Revenue"].dtype == bool else df_raw["Revenue"]
        cnts = r.value_counts()
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.patch.set_facecolor(C["navy2"])

        # Bar
        ax = axes[0]
        ax.set_facecolor(C["navy3"])
        bars = ax.bar(["No Purchase","Purchase"], cnts.values,
                      color=[C["rose"], C["teal"]], width=0.45, edgecolor="none",
                      linewidth=0)
        ax.bar_label(bars, fmt="{:,.0f}", padding=5, color=C["white"], fontsize=11, fontweight="bold")
        ax.set_title("Class Counts", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Sessions")
        ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
        for sp in ax.spines.values(): sp.set_visible(False)

        # Donut
        ax = axes[1]
        ax.set_facecolor(C["navy2"])
        wedges, texts, autos = ax.pie(
            cnts.values, labels=["No Purchase","Purchase"],
            autopct="%1.1f%%",
            colors=[C["rose"], C["teal"]],
            startangle=90,
            wedgeprops={"edgecolor": C["navy2"], "linewidth": 3, "width": 0.6},
        )
        for t in texts:  t.set_color(C["slate"]); t.set_fontsize(11)
        for a in autos:  a.set_color(C["white"]); a.set_fontweight("bold"); a.set_fontsize(11)
        ax.set_title("Class Split", fontsize=13, fontweight="bold", pad=12)

        # Feature variance top 8
        ax = axes[2]
        ax.set_facecolor(C["navy3"])
        num_df = df_raw.select_dtypes(include=np.number)
        top_var = num_df.var().nlargest(8)
        ax.barh(top_var.index[::-1], top_var.values[::-1],
                color=[C["teal"] if i%2==0 else C["teal2"] for i in range(len(top_var))],
                edgecolor="none", height=0.6)
        ax.set_title("Feature Variance (Top 8)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Variance")
        ax.grid(axis="x", alpha=0.3); ax.set_axisbelow(True)
        for sp in ax.spines.values(): sp.set_visible(False)

        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — MODEL LAB
# ══════════════════════════════════════════════════════════════════════════════
with T2:
    chart_style()
    st.markdown(f"<div class='section-eyebrow' style='margin-top:8px'>Training</div><div class='section-title'>Model Lab</div><div class='section-sub'>Train the selected algorithm and inspect its performance across all metrics.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if not has_target:
        st.warning("Upload a labeled dataset (with **Revenue** column) to train models.")
        st.stop()

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run = st.button("⚡ Train Model Now")
    with col_info:
        st.markdown(f"<div class='info-box'><span style='color:{C['slate']};font-size:13px;'>Model: <b style='color:{C['teal']};'>{model_key}</b> &nbsp;·&nbsp; Test split: <b style='color:{C['teal']};'>{test_pct}%</b> &nbsp;·&nbsp; Preprocessing: Label encoding → IQR capping → StandardScaler</span></div>", unsafe_allow_html=True)

    if run:
        with st.spinner("Training in progress…"):
            d, sc = preprocess(df_raw)
            X = d[FEATURE_COLS]
            y = d["Revenue"].astype(int) if d["Revenue"].dtype == bool else d["Revenue"]
            Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=test_pct/100,random_state=42,stratify=y)
            model, metrics, cm, yp, ypr = train_model(Xtr,Xte,ytr,yte,model_key)
        st.session_state.update({"model":model,"metrics":metrics,"cm":cm,
                                  "y_test":yte,"y_pred":yp,"y_proba":ypr,
                                  "scaler":sc,"model_key":model_key})
        st.success(f"✅ {model_key} trained on {len(Xtr):,} samples — {len(Xte):,} held out for evaluation.")

    if "metrics" not in st.session_state:
        st.markdown(f"<div class='info-box' style='text-align:center;padding:32px;color:{C['slate']};'>Hit <b style='color:{C['teal']};'>Train Model Now</b> to see results here.</div>", unsafe_allow_html=True)
    else:
        m   = st.session_state["metrics"]
        cm  = st.session_state["cm"]
        yte = st.session_state["y_test"]
        ypr = st.session_state["y_proba"]
        mk  = st.session_state["model_key"]
        acc_col = C["teal"] if m["Accuracy"] >= 0.85 else C["amber"]

        st.markdown("<br>", unsafe_allow_html=True)
        mc1,mc2,mc3,mc4,mc5 = st.columns(5)
        mc1.metric("🎯 Accuracy",  f"{m['Accuracy']*100:.2f}%")
        mc2.metric("🎖️ Precision", f"{m['Precision']*100:.2f}%")
        mc3.metric("📡 Recall",    f"{m['Recall']*100:.2f}%")
        mc4.metric("⚖️ F1-Score",  f"{m['F1']*100:.2f}%")
        mc5.metric("📈 ROC AUC",   f"{m['ROC AUC']:.4f}" if m["ROC AUC"] else "—")

        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)

        # Confusion matrix
        with ch1:
            st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C['white']};margin-bottom:8px;'>🔲 Confusion Matrix</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4.2))
            fig.patch.set_facecolor(C["navy2"])
            ax.set_facecolor(C["navy3"])
            im = ax.imshow(cm, cmap="YlGn", aspect="auto")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["No Purchase","Purchase"], fontsize=11)
            ax.set_yticklabels(["No Purchase","Purchase"], fontsize=11)
            ax.set_xlabel("Predicted", fontsize=11, labelpad=8)
            ax.set_ylabel("Actual", fontsize=11, labelpad=8)
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,f"{cm[i,j]:,}", ha="center", va="center",
                            fontsize=18, fontweight="bold",
                            color=C["navy"] if cm[i,j] > cm.max()/2 else C["white"])
            ax.set_title(f"{mk.split(' ',1)[1] if ' ' in mk else mk}", fontsize=12, fontweight="bold", pad=12)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # ROC curve
        with ch2:
            st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C['white']};margin-bottom:8px;'>📈 ROC Curve</div>", unsafe_allow_html=True)
            if ypr is not None:
                fpr,tpr,_ = roc_curve(yte, ypr)
                fig, ax = plt.subplots(figsize=(5, 4.2))
                fig.patch.set_facecolor(C["navy2"])
                ax.set_facecolor(C["navy3"])
                ax.fill_between(fpr, tpr, alpha=0.15, color=C["teal"])
                ax.plot(fpr, tpr, color=C["teal"], lw=2.5,
                        label=f"AUC = {m['ROC AUC']:.3f}")
                ax.plot([0,1],[0,1],"--",color=C["slate"],lw=1.5,label="Random")
                ax.set_xlabel("False Positive Rate", fontsize=11)
                ax.set_ylabel("True Positive Rate", fontsize=11)
                ax.set_title("Receiver Operating Characteristic", fontsize=12, fontweight="bold", pad=12)
                ax.legend(loc="lower right", facecolor=C["navy4"],
                          edgecolor=C["border"], labelcolor=C["white"], fontsize=11)
                ax.grid(True, alpha=0.3); ax.set_axisbelow(True)
                for sp in ax.spines.values(): sp.set_color(C["border"])
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

        # Radar / bar comparison
        st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C['white']};margin:18px 0 10px;'>📊 Metric Overview</div>", unsafe_allow_html=True)
        met_names = [k for k,v in m.items() if v is not None]
        met_vals  = [v for v in m.values() if v is not None]
        fig, ax = plt.subplots(figsize=(11,3.2))
        fig.patch.set_facecolor(C["navy2"])
        ax.set_facecolor(C["navy3"])
        x = np.arange(len(met_names))
        bars = ax.bar(met_names, met_vals,
                      color=[C["teal"], C["teal2"], C["amber"], C["violet"], "#06B6D4"],
                      edgecolor="none", width=0.5)
        for bar, v in zip(bars, met_vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                    f"{v:.3f}", ha="center", va="bottom",
                    color=C["white"], fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
        ax.set_title(f"All Metrics — {mk}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
        for sp in ax.spines.values(): sp.set_color(C["border"])
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with T3:
    chart_style()
    st.markdown(f"<div class='section-eyebrow' style='margin-top:8px'>Inference</div><div class='section-title'>Predict on Uploaded Data</div><div class='section-sub'>Runs the trained model on every row and exports results with confidence scores.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if "model" not in st.session_state:
        st.markdown(f"<div class='info-box' style='text-align:center;padding:32px;color:{C['slate']};'>Train a model in <b style='color:{C['teal']};'>Model Lab</b> first.</div>", unsafe_allow_html=True)
    else:
        model  = st.session_state["model"]
        sc     = st.session_state["scaler"]
        mk     = st.session_state["model_key"]

        pred_raw = df_raw[FEATURE_COLS].copy()
        try:
            le = LabelEncoder()
            pred_raw["Month"]       = le.fit_transform(pred_raw["Month"].astype(str))
            pred_raw["VisitorType"] = le.fit_transform(pred_raw["VisitorType"].astype(str))
            pred_raw["Weekend"]     = pred_raw["Weekend"].astype(int)
            for col in OUTLIER_COLS:
                Q1,Q3 = pred_raw[col].quantile(.25), pred_raw[col].quantile(.75)
                IQR = Q3-Q1
                pred_raw[col] = pred_raw[col].clip(Q1-1.5*IQR, Q3+1.5*IQR)
            num = pred_raw.select_dtypes(include=np.number).columns.tolist()
            pred_raw[num] = sc.transform(pred_raw[num])
        except Exception as e:
            st.error(f"Preprocessing error: {e}"); st.stop()

        with st.spinner("Scoring all rows…"):
            preds  = model.predict(pred_raw[FEATURE_COLS])
            probas = model.predict_proba(pred_raw[FEATURE_COLS])[:,1] \
                     if hasattr(model,"predict_proba") else np.full(len(preds), np.nan)

        n_yes = int(preds.sum())
        n_no  = len(preds) - n_yes
        avg_c = probas[~np.isnan(probas)].mean()*100 if not np.all(np.isnan(probas)) else np.nan

        p1,p2,p3,p4 = st.columns(4)
        p1.metric("Total Scored", f"{len(preds):,}")
        p2.metric("Will Purchase",  f"{n_yes:,}", f"{n_yes/len(preds)*100:.1f}%")
        p3.metric("Won't Purchase", f"{n_no:,}",  f"{n_no/len(preds)*100:.1f}%")
        p4.metric("Avg Confidence", f"{avg_c:.1f}%" if not np.isnan(avg_c) else "—")

        # Confidence histogram
        if not np.all(np.isnan(probas)):
            st.markdown("<br>", unsafe_allow_html=True)
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            fig.patch.set_facecolor(C["navy2"])

            ax = axes[0]
            ax.set_facecolor(C["navy3"])
            ax.hist(probas[preds==0]*100, bins=35, color=C["rose"], alpha=0.8,
                    label="No Purchase", edgecolor="none")
            ax.hist(probas[preds==1]*100, bins=35, color=C["teal"], alpha=0.8,
                    label="Purchase", edgecolor="none")
            ax.axvline(50, color=C["amber"], ls="--", lw=1.5, label="Threshold 50%")
            ax.set_xlabel("Confidence (%)"); ax.set_ylabel("Count")
            ax.set_title("Confidence Distribution", fontsize=13, fontweight="bold")
            ax.legend(facecolor=C["navy4"], edgecolor=C["border"], labelcolor=C["white"])
            ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
            for sp in ax.spines.values(): sp.set_color(C["border"])

            ax = axes[1]
            ax.set_facecolor(C["navy3"])
            # cumulative confidence
            sorted_conf = np.sort(probas)
            ax.plot(np.linspace(0,100,len(sorted_conf)), sorted_conf*100,
                    color=C["teal"], lw=2.5)
            ax.fill_between(np.linspace(0,100,len(sorted_conf)), sorted_conf*100,
                            alpha=0.15, color=C["teal"])
            ax.axhline(50, color=C["amber"], ls="--", lw=1.5, label="50% line")
            ax.set_xlabel("Percentile of Samples")
            ax.set_ylabel("Confidence Score (%)")
            ax.set_title("Cumulative Confidence", fontsize=13, fontweight="bold")
            ax.legend(facecolor=C["navy4"], edgecolor=C["border"], labelcolor=C["white"])
            ax.grid(alpha=0.3); ax.set_axisbelow(True)
            for sp in ax.spines.values(): sp.set_color(C["border"])

            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Results table
        st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C['white']};margin:18px 0 8px;'>📋 Prediction Results</div>", unsafe_allow_html=True)
        result_df = df_raw.copy()
        result_df["Prediction"]   = preds
        result_df["Will_Purchase"] = result_df["Prediction"].map({1:"YES ✅",0:"NO ❌"})
        result_df["Confidence_%"] = (probas*100).round(2)
        show_cols = ["Will_Purchase","Confidence_%"] + FEATURE_COLS[:6]
        st.dataframe(result_df[show_cols].rename(columns={"Will_Purchase":"Prediction"}),
                     use_container_width=True, height=320)

        # Download
        buf = io.StringIO()
        result_df.drop(columns=["Will_Purchase"], errors="ignore").to_csv(buf, index=False)
        st.download_button(
            "⬇️  Download Predictions CSV",
            data=buf.getvalue(),
            file_name="shopintent_predictions.csv",
            mime="text/csv",
        )

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — DEEP CHARTS
# ══════════════════════════════════════════════════════════════════════════════
with T4:
    chart_style()
    st.markdown(f"<div class='section-eyebrow' style='margin-top:8px'>Insights</div><div class='section-title'>Deep Charts</div><div class='section-sub'>Rich exploratory visualisations of the dataset.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    num_df = df_raw.select_dtypes(include=np.number).copy()
    if has_target and "Revenue" not in num_df.columns:
        num_df["Revenue"] = df_raw["Revenue"].astype(int)

    # ── Correlation heatmap ──
    st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C['white']};margin-bottom:10px;'>🌡️ Correlation Heatmap</div>", unsafe_allow_html=True)
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14,7))
    fig.patch.set_facecolor(C["navy2"])
    ax.set_facecolor(C["navy3"])
    cmap = sns.diverging_palette(340, 160, s=80, l=45, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                annot=True, fmt=".2f", ax=ax, linewidths=0.6,
                linecolor=C["navy2"], annot_kws={"size": 7.5},
                cbar_kws={"shrink":.8})
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=14)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Box plots ──
    st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C['white']};margin-bottom:10px;'>📦 Outlier Distributions</div>", unsafe_allow_html=True)
    avail = [c for c in OUTLIER_COLS if c in df_raw.columns]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.patch.set_facecolor(C["navy2"])
    axes = axes.flatten()
    pal = [C["teal"], C["amber"], C["violet"], C["rose"], "#06B6D4", C["teal2"]]
    for i, feat in enumerate(avail):
        ax = axes[i]; ax.set_facecolor(C["navy3"])
        bp = ax.boxplot(df_raw[feat].dropna(), patch_artist=True, notch=True,
                        medianprops={"color": C["white"], "linewidth": 2.5},
                        boxprops={"facecolor": pal[i], "alpha": 0.55},
                        whiskerprops={"color": C["slate"], "linewidth": 1.5},
                        capprops={"color": C["slate"], "linewidth": 1.5},
                        flierprops={"marker":"o","markerfacecolor":pal[i],
                                    "markersize":3,"alpha":.4,"markeredgewidth":0})
        ax.set_title(feat.replace("_"," "), fontsize=10, fontweight="600")
        ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
        for sp in ax.spines.values(): sp.set_color(C["border"])
    for j in range(len(avail), len(axes)): axes[j].set_visible(False)
    fig.suptitle("Key Feature Distributions (Before Capping)", fontsize=13,
                 fontweight="bold", color=C["white"], y=1.01)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    two_col = st.columns(2)

    # ── Monthly traffic ──
    with two_col[0]:
        if "Month" in df_raw.columns:
            st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C['white']};margin-bottom:8px;'>📅 Sessions by Month</div>", unsafe_allow_html=True)
            mo = ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]
            mc = df_raw["Month"].value_counts().reindex([m for m in mo if m in df_raw["Month"].unique()]).dropna()
            fig, ax = plt.subplots(figsize=(6.5,4))
            fig.patch.set_facecolor(C["navy2"]); ax.set_facecolor(C["navy3"])
            grad = plt.cm.cool(np.linspace(0.15,0.9,len(mc)))
            bars = ax.bar(mc.index, mc.values, color=grad, edgecolor="none", width=0.65)
            ax.bar_label(bars, fmt="{:,.0f}", padding=3, color=C["white"], fontsize=8.5, fontweight="bold")
            ax.set_xlabel("Month"); ax.set_ylabel("Sessions")
            ax.set_title("Monthly Traffic Volume", fontsize=12, fontweight="bold")
            ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
            plt.xticks(rotation=30, ha="right")
            for sp in ax.spines.values(): sp.set_color(C["border"])
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

    # ── Visitor type purchase rate ──
    with two_col[1]:
        if has_target and "VisitorType" in df_raw.columns:
            st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C['white']};margin-bottom:8px;'>👤 Purchase Rate by Visitor Type</div>", unsafe_allow_html=True)
            r = df_raw["Revenue"].astype(int) if df_raw["Revenue"].dtype == bool else df_raw["Revenue"]
            vt = df_raw.assign(Revenue_int=r).groupby("VisitorType")["Revenue_int"].mean().mul(100).reset_index()
            vt.columns = ["VisitorType","Rate"]
            fig, ax = plt.subplots(figsize=(6.5,4))
            fig.patch.set_facecolor(C["navy2"]); ax.set_facecolor(C["navy3"])
            colors = [C["teal"], C["amber"], C["violet"]]
            bars = ax.bar(vt["VisitorType"], vt["Rate"],
                          color=colors[:len(vt)], edgecolor="none", width=0.45)
            ax.bar_label(bars, fmt="{:.1f}%", padding=3, color=C["white"], fontsize=10, fontweight="bold")
            ax.set_ylabel("Purchase Rate (%)")
            ax.set_title("Purchase Rate by Visitor Type", fontsize=12, fontweight="bold")
            ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
            for sp in ax.spines.values(): sp.set_color(C["border"])
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

    # ── Weekend vs Weekday + Feature importances ──
    two_col2 = st.columns(2)

    with two_col2[0]:
        if has_target and "Weekend" in df_raw.columns:
            st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C['white']};margin:14px 0 8px;'>📆 Weekend vs Weekday</div>", unsafe_allow_html=True)
            r = df_raw["Revenue"].astype(int) if df_raw["Revenue"].dtype == bool else df_raw["Revenue"]
            wk = df_raw.assign(Rev=r).groupby("Weekend")["Rev"].agg(["mean","count"]).reset_index()
            wk["label"] = wk["Weekend"].map({True:"Weekend",False:"Weekday",1:"Weekend",0:"Weekday"})
            fig, ax = plt.subplots(figsize=(6.5,4))
            fig.patch.set_facecolor(C["navy2"]); ax.set_facecolor(C["navy3"])
            bars = ax.bar(wk["label"], wk["mean"]*100,
                          color=[C["teal"], C["amber"]], edgecolor="none", width=0.35)
            ax.bar_label(bars, fmt="{:.1f}%", padding=3, color=C["white"], fontsize=11, fontweight="bold")
            ax.set_ylabel("Purchase Rate (%)"); ax.set_ylim(0, wk["mean"].max()*130)
            ax.set_title("Purchase Rate: Weekend vs Weekday", fontsize=12, fontweight="bold")
            ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
            for sp in ax.spines.values(): sp.set_color(C["border"])
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with two_col2[1]:
        if "model" in st.session_state and hasattr(st.session_state["model"], "feature_importances_"):
            st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C['white']};margin:14px 0 8px;'>🌲 Feature Importances</div>", unsafe_allow_html=True)
            fi = pd.Series(st.session_state["model"].feature_importances_, index=FEATURE_COLS).nlargest(12)
            fig, ax = plt.subplots(figsize=(6.5,4.5))
            fig.patch.set_facecolor(C["navy2"]); ax.set_facecolor(C["navy3"])
            colors_fi = [C["teal"] if i<3 else C["teal2"] if i<6 else C["slate"] for i in range(len(fi))]
            ax.barh(fi.index[::-1], fi.values[::-1], color=colors_fi[::-1], edgecolor="none", height=0.6)
            ax.set_xlabel("Importance")
            ax.set_title("Top Feature Importances", fontsize=12, fontweight="bold")
            ax.grid(axis="x", alpha=0.3); ax.set_axisbelow(True)
            for sp in ax.spines.values(): sp.set_color(C["border"])
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        else:
            st.markdown(f"<div class='info-box' style='margin-top:14px;color:{C['slate']};font-size:13px;'>Train a <b style='color:{C['teal']};'>Random Forest</b> or <b style='color:{C['teal']};'>Decision Tree</b> to see feature importances here.</div>", unsafe_allow_html=True)

    # ── Page values vs Revenue scatter ──
    if has_target and "PageValues" in df_raw.columns:
        st.markdown(f"<div style='font-size:14px;font-weight:600;color:{C['white']};margin:20px 0 8px;'>💎 PageValues vs Exit Rate (coloured by Purchase)</div>", unsafe_allow_html=True)
        r = df_raw["Revenue"].astype(int) if df_raw["Revenue"].dtype == bool else df_raw["Revenue"]
        sample = df_raw.sample(min(2000, len(df_raw)), random_state=42)
        r_s    = r.loc[sample.index]
        fig, ax = plt.subplots(figsize=(12,4.5))
        fig.patch.set_facecolor(C["navy2"]); ax.set_facecolor(C["navy3"])
        ax.scatter(sample.loc[r_s==0,"PageValues"], sample.loc[r_s==0,"ExitRates"],
                   color=C["rose"], alpha=0.35, s=18, label="No Purchase", edgecolors="none")
        ax.scatter(sample.loc[r_s==1,"PageValues"], sample.loc[r_s==1,"ExitRates"],
                   color=C["teal"], alpha=0.65, s=22, label="Purchase", edgecolors="none")
        ax.set_xlabel("PageValues"); ax.set_ylabel("ExitRates")
        ax.set_title("PageValues vs ExitRates — Purchase Segmentation", fontsize=13, fontweight="bold")
        ax.legend(facecolor=C["navy4"], edgecolor=C["border"], labelcolor=C["white"])
        ax.grid(alpha=0.3); ax.set_axisbelow(True)
        for sp in ax.spines.values(): sp.set_color(C["border"])
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
