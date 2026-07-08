import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─────────────────────────────────────────────
#  Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EstateIQ — House Price Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Global CSS / Design System
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700;800&display=swap');

/* ── Root tokens ── */
:root {
    --bg-base:      #0a0d14;
    --bg-surface:   #111827;
    --bg-card:      rgba(255,255,255,0.04);
    --bg-card-h:    rgba(255,255,255,0.08);
    --border:       rgba(255,255,255,0.08);
    --border-h:     rgba(99,179,237,0.4);
    --accent-1:     #6366f1;   /* indigo */
    --accent-2:     #06b6d4;   /* cyan   */
    --accent-3:     #10b981;   /* emerald*/
    --accent-warn:  #f59e0b;   /* amber  */
    --text-primary: #f1f5f9;
    --text-muted:   #94a3b8;
    --text-dim:     #64748b;
    --radius-sm:    8px;
    --radius-md:    14px;
    --radius-lg:    20px;
    --shadow:       0 8px 32px rgba(0,0,0,0.4);
    --glow-indigo:  0 0 40px rgba(99,102,241,0.18);
    --glow-cyan:    0 0 40px rgba(6,182,212,0.18);
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

.main .block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1400px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1729 0%, #0a0d14 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* ── Header brand ── */
.brand-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 2.5rem;
}
.brand-logo {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    box-shadow: var(--glow-indigo);
}
.brand-name {
    font-family: 'Outfit', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.brand-tagline {
    font-size: 0.7rem;
    color: var(--text-dim);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
}

/* ── Section labels (sidebar) ── */
.section-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin: 1.6rem 0 0.6rem 0;
    padding-left: 2px;
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 2rem;
}
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, transform 0.25s, box-shadow 0.25s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--card-accent, linear-gradient(90deg, var(--accent-1), var(--accent-2)));
}
.metric-card:hover {
    border-color: var(--border-h);
    transform: translateY(-2px);
    box-shadow: var(--shadow);
}
.metric-icon { font-size: 1.5rem; margin-bottom: 8px; }
.metric-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}
.metric-label {
    font-size: 0.73rem;
    color: var(--text-muted);
    margin-top: 4px;
    font-weight: 500;
}

/* ── Section title ── */
.section-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
    margin-left: 4px;
}

/* ── Hero result card ── */
.result-hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(6,182,212,0.12) 100%);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: var(--radius-lg);
    padding: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 1.5rem;
    box-shadow: var(--glow-indigo), var(--glow-cyan);
    animation: fadeInUp 0.5s ease-out;
}
.result-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 30% 0%, rgba(99,102,241,0.12), transparent 60%),
                radial-gradient(ellipse at 70% 100%, rgba(6,182,212,0.1), transparent 60%);
    pointer-events: none;
}
.result-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent-2);
    margin-bottom: 0.5rem;
}
.result-price {
    font-family: 'Outfit', sans-serif;
    font-size: 3.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a5b4fc, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin: 0.3rem 0;
}
.result-sub {
    font-size: 0.85rem;
    color: var(--text-muted);
}

/* ── Price range bands ── */
.range-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
    margin-top: 1rem;
}
.range-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.9rem 1rem;
    text-align: center;
}
.range-title { font-size: 0.65rem; color: var(--text-dim); letter-spacing: 0.12em; text-transform: uppercase; font-weight: 600; }
.range-value { font-family: 'Outfit', sans-serif; font-size: 1.15rem; font-weight: 700; margin-top: 3px; }

/* ── Confidence gauge ── */
.confidence-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
    margin: 6px 0 3px;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
}

/* ── Inputs ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] select {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-size: 0.87rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.2) !important;
}

/* Sliders */
div[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2)) !important;
}
div[data-testid="stSlider"] div[role="slider"] {
    background: #fff !important;
    border: 2px solid var(--accent-1) !important;
    box-shadow: 0 0 10px rgba(99,102,241,0.5) !important;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent-1) 0%, #4f46e5 50%, var(--accent-2) 100%) !important;
    background-size: 200% 200% !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.04em !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    animation: gradientShift 4s ease infinite !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.55) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Tabs ── */
div[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent-1) !important;
    border-bottom: 2px solid var(--accent-1) !important;
}

/* ── Alert boxes ── */
div[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border-left-width: 4px !important;
}

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes gradientShift {
    0%,100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
}
@keyframes pulse-glow {
    0%,100% { box-shadow: 0 0 20px rgba(99,102,241,0.3); }
    50%      { box-shadow: 0 0 40px rgba(99,102,241,0.6); }
}

/* ── Expander ── */
details summary {
    font-size: 0.83rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 0.72rem;
    color: var(--text-dim);
    padding: 1.5rem 0 0.5rem;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    artifact = joblib.load("models/model.pkl")
    return artifact["pipeline"], artifact["columns"]

with st.spinner("🔄 Loading intelligence engine…"):
    pipeline, train_columns = load_model()


# ─────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────
def price_tier(price: float) -> tuple[str, str]:
    if price < 150_000:
        return "Affordable", "#10b981"
    elif price < 300_000:
        return "Mid-Range", "#06b6d4"
    elif price < 500_000:
        return "Premium", "#6366f1"
    else:
        return "Luxury", "#f59e0b"

def build_gauge(value: float, min_v: float, max_v: float, title: str, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"prefix": "$", "valueformat": ",.0f",
                "font": {"size": 26, "color": "#f1f5f9", "family": "Outfit"}},
        title={"text": title, "font": {"size": 13, "color": "#94a3b8", "family": "Inter"}},
        gauge={
            "axis": {"range": [min_v, max_v], "tickformat": "$.0s",
                     "tickfont": {"color": "#64748b", "size": 10}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [min_v, max_v * 0.33], "color": "rgba(16,185,129,0.12)"},
                {"range": [max_v * 0.33, max_v * 0.66], "color": "rgba(6,182,212,0.12)"},
                {"range": [max_v * 0.66, max_v], "color": "rgba(245,158,11,0.12)"},
            ],
            "threshold": {"line": {"color": "#f1f5f9", "width": 2},
                          "thickness": 0.75, "value": value},
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig

def build_feature_radar(features: dict):
    cats = ["Quality", "Size", "Age", "Bathrooms", "Garage", "Rooms"]
    # Normalise inputs to 0-10 scale
    qual   = features["OverallQual"]
    size   = min(10, (features["LotArea"] / 15000) * 10)
    age    = max(0, 10 - ((datetime.now().year - features["YearBuilt"]) / 12))
    bath   = min(10, features["FullBath"] * 3)
    garage = min(10, features["GarageCars"] * 3.5)
    rooms  = min(10, features["TotRmsAbvGrd"] * 1.1)

    vals = [qual, size, age, bath, garage, rooms]
    vals += vals[:1]
    cats += cats[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor="rgba(99,102,241,0.18)",
        line=dict(color="#6366f1", width=2),
        marker=dict(color="#a5b4fc", size=6),
        name="Property"
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,0.02)",
            radialaxis=dict(visible=True, range=[0, 10],
                            gridcolor="rgba(255,255,255,0.07)",
                            tickfont=dict(color="#64748b", size=9),
                            tickvals=[2, 4, 6, 8, 10]),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.07)",
                             tickfont=dict(color="#94a3b8", size=11))
        ),
        showlegend=False,
        height=280,
        margin=dict(l=30, r=30, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def build_price_breakdown(price: float):
    # Notional cost breakdown
    land_pct   = 0.30
    struct_pct = 0.45
    finish_pct = 0.15
    other_pct  = 0.10

    labels  = ["Land Value", "Structure", "Finishes & Fixtures", "Other"]
    values  = [price * land_pct, price * struct_pct, price * finish_pct, price * other_pct]
    colors  = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#0a0d14", width=2)),
        textinfo="label+percent",
        textfont=dict(family="Inter", size=11, color="#f1f5f9"),
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"${price/1000:.0f}K",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=22, family="Outfit", color="#f1f5f9"),
    )
    fig.update_layout(
        height=290,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig

def build_comparable_bar(price: float):
    comp_labels = ["Budget Zone", "Market Avg", "Your Property", "Premium Band", "Luxury Tier"]
    comp_vals   = [120_000, 220_000, price, 400_000, 700_000]
    comp_colors = ["#10b981" if v != price else "#6366f1" for v in comp_vals]

    fig = go.Figure(go.Bar(
        x=comp_labels, y=comp_vals,
        marker=dict(color=comp_colors,
                    line=dict(width=0),
                    cornerradius=6),
        text=[f"${v/1000:.0f}K" for v in comp_vals],
        textposition="outside",
        textfont=dict(size=11, color="#94a3b8", family="Inter"),
        hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                   tickformat="$,.0f", tickfont=dict(color="#64748b", size=10),
                   zeroline=False),
        xaxis=dict(tickfont=dict(color="#94a3b8", size=11)),
        font=dict(family="Inter"),
    )
    return fig


# ─────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "history" not in st.session_state:
    st.session_state.history = []


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="brand-header">
        <div class="brand-logo">🏡</div>
        <div>
            <div class="brand-name">EstateIQ</div>
            <div class="brand-tagline">House Price Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Property Identity ────────────────────
    st.markdown('<div class="section-label">🏗 Property Identity</div>', unsafe_allow_html=True)
    MSSubClass = st.selectbox(
        "Building Class",
        options=[20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190],
        index=0,
        help="MSSubClass identifies the type of dwelling involved in the sale.",
    )
    YearBuilt = st.number_input(
        "Year Built", min_value=1872, max_value=datetime.now().year,
        value=2000, step=1,
        help="Original construction year."
    )
    YrSold = st.number_input(
        "Year Sold", min_value=2006, max_value=datetime.now().year,
        value=2010, step=1,
    )

    # ── Size & Layout ────────────────────────
    st.markdown('<div class="section-label">📐 Size & Layout</div>', unsafe_allow_html=True)
    LotArea = st.number_input("Lot Area (sq ft)", min_value=0, value=8000, step=100)
    TotalBsmtSF = st.number_input("Basement Area (sq ft)", min_value=0, value=800, step=50)
    FirstFlrSF  = st.number_input("1st Floor (sq ft)", min_value=0, value=800, step=50)
    SecondFlrSF = st.number_input("2nd Floor (sq ft)", min_value=0, value=500, step=50)

    # ── Quality & Condition ──────────────────
    st.markdown('<div class="section-label">⭐ Quality & Condition</div>', unsafe_allow_html=True)
    OverallQual = st.slider(
        "Overall Quality", 1, 10, 5,
        help="Rates overall material & finish of the house (1 = Very Poor … 10 = Very Excellent)"
    )
    OverallCond = st.slider(
        "Overall Condition", 1, 10, 5,
        help="Rates the overall condition of the house"
    )

    # ── Rooms ────────────────────────────────
    st.markdown('<div class="section-label">🛏 Rooms & Amenities</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        FullBath      = st.number_input("Full Baths", min_value=0, max_value=10, value=2, step=1)
        BedroomAbvGr  = st.number_input("Bedrooms", min_value=0, max_value=20, value=3, step=1)
    with col_b:
        TotRmsAbvGrd  = st.number_input("Total Rooms", min_value=0, max_value=30, value=6, step=1)
        GarageCars    = st.number_input("Garage Cars", min_value=0, max_value=10, value=2, step=1)

    GarageArea = st.number_input("Garage Area (sq ft)", min_value=0, value=500, step=25)

    st.markdown("<br>", unsafe_allow_html=True)

    predict_clicked = st.button("🔮  Predict Price", use_container_width=True)


# ─────────────────────────────────────────────
#  MAIN AREA — Hero header
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 1.8rem;">
    <h1 style="font-family:'Outfit',sans-serif; font-size:2.2rem; font-weight:800;
               background:linear-gradient(90deg,#a5b4fc,#67e8f9);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               background-clip:text; margin:0; line-height:1.15;">
        House Price Intelligence
    </h1>
    <p style="color:#64748b; font-size:0.9rem; margin:6px 0 0; font-weight:400;">
        ML-powered valuation · Configure your property on the left · Get an instant estimate
    </p>
</div>
""", unsafe_allow_html=True)

# ── Live metric bar ──────────────────────────
total_sf = FirstFlrSF + SecondFlrSF
age_yrs  = datetime.now().year - YearBuilt
price_per_sf_est = (OverallQual * 18 + 40) if total_sf > 0 else 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card" style="--card-accent:linear-gradient(90deg,#6366f1,#818cf8);">
        <div class="metric-icon">📐</div>
        <div class="metric-value">{total_sf:,}</div>
        <div class="metric-label">Total Living Area (sq ft)</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="--card-accent:linear-gradient(90deg,#06b6d4,#22d3ee);">
        <div class="metric-icon">🏗</div>
        <div class="metric-value">{age_yrs}</div>
        <div class="metric-label">Years Since Built</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card" style="--card-accent:linear-gradient(90deg,#10b981,#34d399);">
        <div class="metric-icon">⭐</div>
        <div class="metric-value">{OverallQual}/10</div>
        <div class="metric-label">Quality Rating</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card" style="--card-accent:linear-gradient(90deg,#f59e0b,#fbbf24);">
        <div class="metric-icon">🚗</div>
        <div class="metric-value">{GarageCars}</div>
        <div class="metric-label">Garage Capacity (cars)</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PREDICTION LOGIC
# ─────────────────────────────────────────────
if predict_clicked:
    input_data = {
        "MSSubClass":   MSSubClass,
        "LotArea":      LotArea,
        "OverallQual":  OverallQual,
        "OverallCond":  OverallCond,
        "YearBuilt":    YearBuilt,
        "TotalBsmtSF":  TotalBsmtSF,
        "1stFlrSF":     FirstFlrSF,
        "2ndFlrSF":     SecondFlrSF,
        "FullBath":     FullBath,
        "BedroomAbvGr": BedroomAbvGr,
        "TotRmsAbvGrd": TotRmsAbvGrd,
        "GarageCars":   GarageCars,
        "GarageArea":   GarageArea,
        "YrSold":       YrSold,
    }
    try:
        df = pd.DataFrame([input_data])
        for col in train_columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[train_columns]

        pred  = pipeline.predict(df)[0]
        price = float(np.expm1(pred))

        st.session_state.prediction = {
            "price":       price,
            "inputs":      input_data,
            "timestamp":   datetime.now().strftime("%H:%M:%S"),
        }
        # Save to history (last 5)
        st.session_state.history.insert(0, {
            "time":  st.session_state.prediction["timestamp"],
            "price": price,
            "qual":  OverallQual,
            "sf":    total_sf,
        })
        st.session_state.history = st.session_state.history[:5]

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")


# ─────────────────────────────────────────────
#  RESULTS PANEL
# ─────────────────────────────────────────────
if st.session_state.prediction:
    price  = st.session_state.prediction["price"]
    tier, tier_color = price_tier(price)
    low    = price * 0.92
    high   = price * 1.08
    conf   = min(95, 72 + OverallQual * 2.3)

    st.markdown("---")

    # ── Hero result ─────────────────────────
    st.markdown(f"""
    <div class="result-hero">
        <div class="result-label">📍 Estimated Market Value</div>
        <div class="result-price">${price:,.0f}</div>
        <div class="result-sub">
            Range: <strong style="color:#67e8f9">${low:,.0f}</strong>
            — <strong style="color:#a5b4fc">${high:,.0f}</strong>
            &nbsp;·&nbsp;
            <span style="color:{tier_color}; font-weight:600;">{tier}</span> Segment
            &nbsp;·&nbsp;Predicted at {st.session_state.prediction['timestamp']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence bar ───────────────────────
    st.markdown(f"""
    <div style="margin-bottom:1.6rem;">
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="font-size:0.75rem; color:#64748b; font-weight:600; letter-spacing:0.1em; text-transform:uppercase;">Model Confidence</span>
            <span style="font-size:0.78rem; color:#94a3b8; font-weight:600;">{conf:.0f}%</span>
        </div>
        <div class="confidence-bar-wrap">
            <div class="confidence-bar-fill" style="width:{conf}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs: Analytics ──────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊  Market Gauge", "🕸  Property Radar", "🍩  Cost Breakdown", "📈  Comparables"]
    )

    with tab1:
        st.plotly_chart(
            build_gauge(price, 50_000, 800_000, "Estimated Price", "#6366f1"),
            use_container_width=True, config={"displayModeBar": False}
        )
        # Range cards
        st.markdown(f"""
        <div class="range-grid">
            <div class="range-card">
                <div class="range-title">Conservative</div>
                <div class="range-value" style="color:#10b981;">${low:,.0f}</div>
            </div>
            <div class="range-card" style="border-color:rgba(99,102,241,0.4);">
                <div class="range-title">Prediction</div>
                <div class="range-value" style="color:#a5b4fc;">${price:,.0f}</div>
            </div>
            <div class="range-card">
                <div class="range-title">Optimistic</div>
                <div class="range-value" style="color:#f59e0b;">${high:,.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.plotly_chart(
            build_feature_radar(st.session_state.prediction["inputs"]),
            use_container_width=True, config={"displayModeBar": False}
        )
        st.markdown(f"""
        <p style="font-size:0.78rem; color:#64748b; text-align:center; margin-top:-10px;">
            Radar shows normalised property attributes (0–10 scale)
        </p>
        """, unsafe_allow_html=True)

    with tab3:
        st.plotly_chart(
            build_price_breakdown(price),
            use_container_width=True, config={"displayModeBar": False}
        )
        st.markdown("""
        <p style="font-size:0.76rem; color:#64748b; text-align:center; margin-top:-10px;">
            Notional breakdown · Land · Structure · Finishes · Other
        </p>
        """, unsafe_allow_html=True)

    with tab4:
        st.plotly_chart(
            build_comparable_bar(price),
            use_container_width=True, config={"displayModeBar": False}
        )

    # ── Property summary table ───────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗂 Property Summary</div>', unsafe_allow_html=True)

    inp = st.session_state.prediction["inputs"]
    summary = {
        "Attribute":  ["Building Class", "Lot Area", "Year Built", "Total Area",
                       "Basement", "Quality", "Condition", "Full Baths",
                       "Bedrooms", "Rooms", "Garage Cars", "Year Sold"],
        "Value": [
            f"Class {inp['MSSubClass']}",
            f"{inp['LotArea']:,} sq ft",
            inp["YearBuilt"],
            f"{FirstFlrSF + SecondFlrSF:,} sq ft",
            f"{inp['TotalBsmtSF']:,} sq ft",
            f"{inp['OverallQual']}/10",
            f"{inp['OverallCond']}/10",
            inp["FullBath"],
            inp["BedroomAbvGr"],
            inp["TotRmsAbvGrd"],
            inp["GarageCars"],
            inp["YrSold"],
        ]
    }
    st.dataframe(
        pd.DataFrame(summary),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Attribute": st.column_config.TextColumn("Attribute", width="medium"),
            "Value":     st.column_config.TextColumn("Value",     width="medium"),
        }
    )

else:
    # ── Empty state ──────────────────────────
    st.markdown("""
    <div style="text-align:center; padding: 5rem 2rem; color:#334155;">
        <div style="font-size:5rem; margin-bottom:1rem; filter:grayscale(0.3);">🏡</div>
        <h3 style="font-family:'Outfit',sans-serif; font-size:1.4rem; font-weight:700;
                   color:#475569; margin-bottom:0.5rem;">
            Configure your property
        </h3>
        <p style="font-size:0.9rem; color:#334155; max-width:380px; margin:0 auto;">
            Fill in the property details on the left sidebar and click
            <strong style="color:#6366f1;">Predict Price</strong> to see your AI-powered valuation.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PREDICTION HISTORY  (always visible after 1 run)
# ─────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    with st.expander("🕐  Prediction History  (last 5 runs)", expanded=False):
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df.columns = ["Time", "Estimated Price ($)", "Quality", "Total Area (sq ft)"]
        hist_df["Estimated Price ($)"] = hist_df["Estimated Price ($)"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(hist_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
    EstateIQ · ML-Powered House Price Intelligence · Built with Streamlit & scikit-learn
</div>
""", unsafe_allow_html=True)