import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
sys.path.append(str(Path(__file__).parent))
from Loyalty_model import LoyaltyEngine, PSYCH_FEATURES, SEGMENT_COLORS
 
# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Loyalty Advisor",
    page_icon="🧠",
    layout="wide",
)
 
# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0e0e1a; }
    [data-testid="stSidebar"]          { background: #12121f; }
 
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #3a3a5c;
        border-radius: 14px;
        padding: 22px 18px;
        text-align: center;
        height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card .label { color: #888; font-size: 0.8rem; margin-bottom: 6px; }
    .metric-card .value { font-size: 1.7rem; font-weight: 700; margin: 0; }
 
    .seg-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
    }
 
    .rec-card {
        background: #161626;
        border-left: 4px solid #A784B6;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
        line-height: 1.7;
    }
    .rec-tag {
        display: inline-block;
        background: #A784B6;
        color: #fff;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 9px;
        border-radius: 4px;
        margin-bottom: 5px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .rec-body { color: #ccc; font-size: 0.93rem; }
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# Segment colors
# ─────────────────────────────────────────────────────────────────────────────
SEG_HEX = {
    "Champions":           "#4CAF50",
    "Loyal Customers":     "#8BC34A",
    "Big Spenders":        "#42A5F5",
    "Potential Loyalists": "#FFB300",
    "Regular Customers":   "#9E9E9E",
    "At Risk / Low Value": "#C44536",
}
 
# ─────────────────────────────────────────────────────────────────────────────
# Load model (cached — loads only once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def load_engine() -> LoyaltyEngine:
    return LoyaltyEngine.load("models/")
 
try:
    engine = load_engine()
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error loading model: {e}")
    st.stop()
 
# ─────────────────────────────────────────────────────────────────────────────
# Feature display metadata
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_META = {
    "stress_level":            ("😰", "Stress Level",            "Higher = worse"),
    "anxiety_score":           ("😟", "Anxiety Score",           "Higher = worse"),
    "self_esteem":             ("💪", "Self Esteem",             "Higher = better"),
    "impulsiveness":           ("⚡", "Impulsiveness",           "Higher = more impulsive"),
    "optimism_score":          ("🌟", "Optimism",               "Higher = better"),
    "life_satisfaction":       ("😊", "Life Satisfaction",       "Higher = better"),
    "social_media_dependency": ("📱", "Social Media Dependency", "Higher = more dependent"),
}
 
NEGATIVE_FEATS = {"stress_level", "anxiety_score", "social_media_dependency"}
POSITIVE_FEATS = {"self_esteem", "optimism_score", "life_satisfaction"}
 
# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — Input form
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Customer Profile")
    st.caption("Enter psychological assessment scores (scale: 0 – 10)")
    st.divider()
 
    profile: dict[str, float] = {}
    for feat in PSYCH_FEATURES:
        icon, label, hint = FEATURE_META.get(feat, ("", feat.replace("_", " ").title(), ""))
        profile[feat] = st.slider(
            f"{icon} {label}",
            min_value=0.0, max_value=10.0,
            value=5.0, step=0.5,
            help=hint,
        )
 
    st.divider()
    analyze_btn = st.button(
        "🔍 Analyze Customer",
        type="primary",
        use_container_width=True,
    )
 
# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 🧠 Customer Loyalty Advisor")
st.caption(
    "Predicts customer segment from psychological profile and generates "
    "personalized strategies to convert them into loyal customers."
)
st.divider()
 
if not analyze_btn:
    st.info("👈 Adjust the customer profile in the sidebar, then click **Analyze Customer**.")
    st.stop()
 
# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Running prediction..."):
    result = engine.predict(profile)
 
seg        = result["segment"]
champ_prob = result["champion_prob"]
exp_spend  = result["expected_spend"]
seg_probs  = result["segment_probs"]
recs       = result["recommendations"]
seg_color  = SEG_HEX.get(seg, "#A784B6")
 
# ─────────────────────────────────────────────────────────────────────────────
# Row 1 — Key metrics
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
 
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Predicted Segment</div>
        <div class="value">
            <span class="seg-badge"
                style="background:{seg_color}22;color:{seg_color};border:1px solid {seg_color};">
                {seg}
            </span>
        </div>
    </div>""", unsafe_allow_html=True)
 
with c2:
    g_color = "#4CAF50" if champ_prob > 0.4 else "#FFB300" if champ_prob > 0.2 else "#C44536"
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Champion Probability</div>
        <div class="value" style="color:{g_color};">{champ_prob:.0%}</div>
    </div>""", unsafe_allow_html=True)
 
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Expected Total Spend</div>
        <div class="value" style="color:#A784B6;">{exp_spend:,.0f}</div>
    </div>""", unsafe_allow_html=True)
 
with c4:
    # Count significant gaps from Champions
    gap_count = sum(
        1 for f in PSYCH_FEATURES
        if abs(profile[f] - engine.champions_profile.get(f, 5.0)) > 2.5
    )
    g_color2 = "#C44536" if gap_count >= 3 else "#FFB300" if gap_count >= 1 else "#4CAF50"
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Large Gaps vs Champions</div>
        <div class="value" style="color:{g_color2};">{gap_count} / {len(PSYCH_FEATURES)}</div>
    </div>""", unsafe_allow_html=True)
 
st.write("")
 
# ─────────────────────────────────────────────────────────────────────────────
# Row 2 — Radar + Bar chart
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1.4, 1])
 
with col_left:
    st.markdown("#### 📊 Psychological Profile vs Champions")
 
    champ_vals   = [engine.champions_profile.get(f, 5.0) for f in PSYCH_FEATURES]
    cust_vals    = [profile[f] for f in PSYCH_FEATURES]
    radar_labels = [FEATURE_META[f][1] for f in PSYCH_FEATURES]
 
    # Close polygon
    rl = radar_labels + [radar_labels[0]]
    cv = cust_vals    + [cust_vals[0]]
    hv = champ_vals   + [champ_vals[0]]
 
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=hv, theta=rl, fill="toself", name="Champions Avg",
        line=dict(color="#4CAF50", width=2),
        fillcolor="rgba(76,175,80,0.10)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=cv, theta=rl, fill="toself", name="This Customer",
        line=dict(color="#A784B6", width=2.5),
        fillcolor="rgba(167,132,182,0.16)",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#12121e",
            radialaxis=dict(visible=True, range=[0, 10],
                            gridcolor="#2a2a3a", tickfont=dict(color="#777", size=9)),
            angularaxis=dict(gridcolor="#2a2a3a", tickfont=dict(color="#bbb", size=10)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="#12121e", bordercolor="#2a2a3a", font=dict(color="#ccc")),
        margin=dict(t=10, b=10, l=10, r=10),
        height=360,
    )
    st.plotly_chart(fig_radar, use_container_width=True)
 
with col_right:
    st.markdown("#### 🎯 Segment Probabilities")
 
    sorted_probs = sorted(seg_probs.items(), key=lambda x: -x[1])
    s_names = [s for s, _ in sorted_probs]
    s_vals  = [p * 100 for _, p in sorted_probs]
    s_cols  = [SEG_HEX.get(s, "#A784B6") for s in s_names]
 
    fig_bar = go.Figure(go.Bar(
        x=s_vals, y=s_names, orientation="h",
        marker=dict(color=s_cols, opacity=0.85),
        text=[f"{v:.1f}%" for v in s_vals],
        textposition="outside",
        textfont=dict(color="#ccc", size=11),
    ))
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#12121e",
        font=dict(color="#ccc"),
        xaxis=dict(range=[0, 115], showgrid=False, showticklabels=False),
        yaxis=dict(gridcolor="#222", tickfont=dict(size=11)),
        margin=dict(t=10, b=10, l=10, r=70),
        height=360,
    )
    st.plotly_chart(fig_bar, use_container_width=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# Row 3 — Feature Gap Analysis
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### 📐 Feature Gap Analysis")
st.caption("Distance between this customer's scores and the Champions average")
 
gap_rows = []
for feat in PSYCH_FEATURES:
    icon, label, _ = FEATURE_META[feat]
    cval   = profile[feat]
    champ  = engine.champions_profile.get(feat, 5.0)
    gap    = round(cval - champ, 2)
    is_neg = feat in NEGATIVE_FEATS
    # For negative features, being ABOVE champions is actually WORSE
    concern = (gap > 0 and is_neg) or (gap < 0 and feat in POSITIVE_FEATS)
    status  = ("🔴 Large concern" if abs(gap) >= 2.5 and concern
               else "⚠️ Moderate gap" if abs(gap) >= 1.0 and concern
               else "✅ On track")
    gap_rows.append({
        "Feature":         f"{icon} {label}",
        "Customer":        cval,
        "Champions Avg":   round(champ, 2),
        "Gap":             gap,
        "Status":          status,
    })
 
gap_df = pd.DataFrame(gap_rows)
 
def _style_gap(val):
    try:
        v = float(val)
        if abs(v) < 1.0:  return "color:#4CAF50; font-weight:600"
        if abs(v) < 2.5:  return "color:#FFB300; font-weight:600"
        return "color:#C44536; font-weight:600"
    except:
        return ""
 
# .map() replaces deprecated .applymap() in pandas >= 2.1
st.dataframe(
    gap_df.style.map(_style_gap, subset=["Gap"]),
    use_container_width=True,
    hide_index=True,
    height=290,
)
 
# ─────────────────────────────────────────────────────────────────────────────
# Row 4 — Recommendations
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("#### 💡 Personalized Retention Recommendations")
st.caption("Evidence-based strategies to convert this customer into a loyal Champion")
 
for rec in recs:
    if rec.startswith("[") and "]" in rec:
        tag, body = rec[1:].split("]", 1)
        body = body.strip()
    else:
        tag, body = "Recommendation", rec
 
    st.markdown(f"""
    <div class="rec-card">
        <span class="rec-tag">{tag}</span><br>
        <span class="rec-body">{body}</span>
    </div>""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
col_f1, col_f2 = st.columns([3, 1])
with col_f1:
    st.caption("🤖 Powered by Gradient Boosting — trained on psychological & behavioral data")
with col_f2:
    feats_str = " · ".join(PSYCH_FEATURES)
    st.caption(f"Features: {len(PSYCH_FEATURES)}")
 