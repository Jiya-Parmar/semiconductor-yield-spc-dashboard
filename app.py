"""
=============================================================================
  Semiconductor Yield & WAT Analysis Dashboard
  Analog Devices Inc. — Yield/Manufacturing Engineering Group
  Gandhinagar, Gujarat
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Semiconductor Yield & SPC Dashboard",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    color: #e6f1ff;
}

.stApp {
    background: #080c14;
}

/* Headings */
h1, h2, h3, h4 {
    color: #22d3ee !important;
    letter-spacing: 0.03em;
}

/* General text */
p, span, div {
    color: #cbd5e1;
}

/* Subtext */
.metric-label {
    font-size: 12px;
    color: #9fb3c8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Metric values */
.metric-value {
    font-size: 30px;
    font-weight: 700;
    color: #22d3ee;
}

/* Cards */
.metric-box {
    background: #0f172a;
    border: 1px solid #1f2a44;
    border-radius: 12px;
    padding: 18px;
}

/* Alerts */
.alert-box {
    background: rgba(255, 80, 80, 0.12);
    border: 1px solid #ff5a5a;
    color: #ff9a9a;
}

.ok-box {
    background: rgba(0, 220, 120, 0.10);
    border: 1px solid #22c55e;
    color: #86efac;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #060a12;
    border-right: 1px solid #1f2a44;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 14px;
    font-weight: 600;
    color: #9fb3c8;
}

.stTabs [aria-selected="true"] {
    color: #22d3ee !important;
}

/* Metrics container */
div[data-testid="metric-container"] {
    background: #0f172a;
    border: 1px solid #1f2a44;
    border-radius: 10px;
    padding: 12px;
}

	
/* Dropdown main box */
div[data-baseweb="select"] > div {
    background-color: #0f172a !important;
    border: 1px solid #334155 !important;
    color: #e6f1ff !important;
}

/* Dropdown text (selected item) */
div[data-baseweb="select"] span {
    color: #e6f1ff !important;
}

/* Dropdown menu items */
ul[role="listbox"] {
    background-color: #0f172a !important;
}

/* Each option */
ul[role="listbox"] li {
    color: #e6f1ff !important;
    background-color: #0f172a !important;
}

/* Hover effect */
ul[role="listbox"] li:hover {
    background-color: #1e293b !important;
    color: #22d3ee !important;
}

/* Selected option */
[aria-selected="true"] {
    background-color: #1e293b !important;
    color: #22d3ee !important;
}

/* Labels like SPC Parameter */
label, .stSelectbox label {
    color: #cbd5e1 !important;
    font-weight: 500;
}

/* Input boxes */
input, .stNumberInput input {
    color: #e6f1ff !important;
    background-color: #0f172a !important;
}

</style>
""", unsafe_allow_html=True)


# ─── Data Generation ──────────────────────────────────────────────────────────
@st.cache_data
def generate_fab_data(n_lots: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    Simulates realistic semiconductor fab data:
    - WAT parameters: Vt, Idsat, Ioff, Sheet Resistance, Contact Resistance
    - Inline defect density
    - Wafer yield using Murphy's model
    - Embedded excursion events at known lots
    """
    rng = np.random.default_rng(seed)

    EXCURSION_LOTS = {28, 29, 30, 57, 58, 59, 73, 74, 75, 100, 101}
    records = []

    for i in range(n_lots):
        exc = i in EXCURSION_LOTS

        for w in range(25):  # 25 wafers per lot
            # --- WAT Parameters (with excursion shifts) ---
            vt      = rng.normal(0.450 + (0.070 if exc else 0.0),  0.014)
            idsat   = rng.normal(482.0 - (58.0  if exc else 0.0),  11.0)
            ioff    = rng.lognormal(np.log(1e-9), 0.25) * (9 if exc else 1)
            rs      = rng.normal(98.0  + (14.0  if exc else 0.0),  3.8)
            cr      = rng.normal(48.0  + (11.0  if exc else 0.0),  2.3)

            # --- Defect density & yield ---
            d0      = rng.gamma(2.0, 0.045) + (0.55 if exc else 0.0)
            chip_a  = 1.5                          # chip area cm²
            y_raw   = np.exp(-d0 * chip_a) * 100  # Murphy's model
            y_meas  = float(np.clip(y_raw + rng.normal(0, 1.2), 0, 100))

            records.append({
                "lot_id":            f"LOT{i+1:04d}",
                "lot_index":         i,
                "wafer":             w + 1,
                "is_excursion":      exc,
                "vt_nmos":           round(float(vt),    4),
                "idsat_nmos":        round(float(idsat),  2),
                "ioff_nmos":         round(float(ioff),  13),
                "sheet_resistance":  round(float(rs),     3),
                "contact_resistance":round(float(cr),     3),
                "defect_density":    round(float(d0),     4),
                "yield_pct":         round(y_meas,        2),
                "process_step":      rng.choice(
                    ["GATE_OX", "POLY_ETCH", "S/D_IMP", "MET1", "MET2"],
                    p=[0.15, 0.25, 0.30, 0.20, 0.10],
                ),
            })

    return pd.DataFrame(records)


@st.cache_data
def lot_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["lot_id", "lot_index", "is_excursion"])
        .agg(
            mean_yield=("yield_pct",           "mean"),
            mean_vt   =("vt_nmos",             "mean"),
            mean_idsat=("idsat_nmos",           "mean"),
            mean_ioff =("ioff_nmos",            "mean"),
            mean_rs   =("sheet_resistance",     "mean"),
            mean_cr   =("contact_resistance",   "mean"),
            mean_d0   =("defect_density",       "mean"),
        )
        .reset_index()
        .sort_values("lot_index")
    )
    return summary


# ─── Helper Functions ─────────────────────────────────────────────────────────
def compute_cpk(values: np.ndarray, lsl: float, usl: float) -> float:
    mu  = np.mean(values)
    sig = np.std(values, ddof=1)
    if sig == 0:
        return 0.0
    cpu = (usl - mu) / (3 * sig)
    cpl = (mu - lsl) / (3 * sig)
    return round(min(cpu, cpl), 3)


def xbar_r_limits(data: np.ndarray, n: int = 5):
    """Compute X-bar & R chart control limits for subgroup size n."""
    # Constants for n=5
    A2, D3, D4 = 0.577, 0.0, 2.114
    groups  = [data[i : i + n] for i in range(0, len(data) - n + 1, n)]
    xbars   = np.array([g.mean() for g in groups])
    ranges  = np.array([g.max() - g.min() for g in groups])
    x2      = xbars.mean()
    r2      = ranges.mean()
    return (xbars, ranges, x2, r2,
            x2 + A2 * r2,   # UCL_x
            x2 - A2 * r2,   # LCL_x
            D4 * r2,         # UCL_r
            D3 * r2)         # LCL_r


def cusum(data: np.ndarray, k_mult: float = 0.5, h_mult: float = 5.0):
    mu = data.mean()
    k  = k_mult * data.std()
    h  = h_mult * data.std()
    cp = np.zeros(len(data))
    cn = np.zeros(len(data))
    for i in range(1, len(data)):
        cp[i] = max(0, cp[i-1] + data[i] - mu - k)
        cn[i] = min(0, cn[i-1] + data[i] - mu + k)
    return cp, cn, h


DARK = "plotly_dark"
ACCENT = "#22d3ee"
WARN   = "#ffa040"
DANGER = "#ff3c3c"
OK     = "#00dc78"
TEXT_PRIMARY = "#e6f1ff"
TEXT_SECONDARY = "#9fb3c8"
CARD_BG = "#0f172a"
BORDER = "#1f2a44"


# ─── Load Data ────────────────────────────────────────────────────────────────
df  = generate_fab_data()
lot = lot_level_summary(df)

PARAM_META = {
    "vt_nmos":            ("Vt NMOS (V)",           "mean_vt",    0.40,  0.50),
    "idsat_nmos":         ("Idsat NMOS (µA/µm)",     "mean_idsat", 430.0, 530.0),
    "sheet_resistance":   ("Sheet Resistance (Ω/□)", "mean_rs",    85.0,  115.0),
    "contact_resistance": ("Contact Resistance (Ω)", "mean_cr",    42.0,  58.0),
}


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Yield Analytics Dashboard")
    st.markdown("**Semiconductor Process Analytics**")
    st.divider()

    selected_key = st.selectbox(
        "SPC Parameter",
        list(PARAM_META.keys()),
        format_func=lambda k: PARAM_META[k][0],
    )
    label, lot_col, default_lsl, default_usl = PARAM_META[selected_key]

    st.markdown("#### Spec Limits")
    lsl = st.number_input("LSL", value=default_lsl, format="%.4f")
    usl = st.number_input("USL", value=default_usl, format="%.4f")

    st.divider()
    st.caption(f"📦 Lots simulated: {lot.shape[0]}")
    st.caption(f"🔬 Wafers simulated: {df.shape[0]}")
    st.caption("Model: Murphy's Yield  |  SPC: X-bar/R + CUSUM")
    st.caption("ML: PCA + Isolation Forest")
    st.divider()
    st.caption("*Project by Jiya Parmar - Semiconductor Analytics Project*")


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#00ccff; margin-bottom:0;'>🔬 Semiconductor Yield & WAT Analysis Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#9fb3c8; font-size:14px; margin-top:2px;'>"
    "Semiconductor Manufacturing · Yield & Process Control Analytics",
    unsafe_allow_html=True,
)
st.divider()


# ─── KPI Row ──────────────────────────────────────────────────────────────────
param_vals   = lot[lot_col].values
cpk_overall  = compute_cpk(param_vals, lsl, usl)
avg_yield    = lot["mean_yield"].mean()
exc_count    = int(lot["is_excursion"].sum())
total_lots   = len(lot)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("📦 Total Lots",  total_lots)
with k2:
    delta = f"{avg_yield - 85:.1f}% vs 85% target"
    st.metric("📊 Avg Yield",   f"{avg_yield:.1f}%",   delta=delta)
with k3:
    st.metric("⚠️ Excursion Lots", exc_count)
with k4:
    cpk_delta = "✅ Capable" if cpk_overall >= 1.33 else "⚠ Below 1.33"
    st.metric("🎯 Cpk",        cpk_overall,             delta=cpk_delta)
with k5:
    st.metric("🔬 Total Wafers", len(df))

st.divider()


# ─── Tabs ─────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    "📈  SPC Charts",
    "📉  Yield Analysis",
    "🔍  Root Cause & PCA",
    "📊  Cpk & Parameter Health",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SPC CHARTS
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    st.subheader(f"Statistical Process Control — {label}")

    data_arr = lot[lot_col].values
    xbars, ranges, xmean, rmean, ucl_x, lcl_x, ucl_r, lcl_r = xbar_r_limits(data_arr)
    n_sg = len(xbars)

    # ── X-bar chart ──────────────────────────────────────────────────────────
    xbar_colors = [DANGER if (v > ucl_x or v < lcl_x) else ACCENT for v in xbars]

    fig_x = go.Figure()
    fig_x.add_trace(go.Scatter(
        x=list(range(n_sg)), y=xbars,
        mode="lines+markers",
        line=dict(color=ACCENT, width=1.5),
        marker=dict(color=xbar_colors, size=7, symbol="circle"),
        name="X̄ (subgroup mean)",
    ))
    for y, col, txt in [
        (ucl_x, DANGER, "UCL"),
        (lcl_x, DANGER, "LCL"),
        (xmean, OK,     "CL"),
        (usl,   WARN,   "USL"),
        (lsl,   WARN,   "LSL"),
    ]:
        fig_x.add_hline(y=y, line=dict(color=col, dash="dash", width=1.5),
                        annotation_text=txt, annotation_position="right",
                        annotation_font_color=col)

    # shade excursion subgroups
    exc_flags = lot["is_excursion"].values
    for sg in range(n_sg):
        lot_i = sg * 5
        if lot_i < len(exc_flags) and exc_flags[lot_i]:
            fig_x.add_vrect(x0=sg - 0.5, x1=sg + 0.5,
                            fillcolor=DANGER, opacity=0.08, line_width=0)

    fig_x.update_layout(
        title="X-bar Chart (Subgroup Means, n=5)",
        xaxis_title="Subgroup",
        yaxis_title=label,
        template=DARK, height=370,
        plot_bgcolor="#0d1526", paper_bgcolor="#080c14",
    )
    st.plotly_chart(fig_x, use_container_width=True)

    # ── R chart ───────────────────────────────────────────────────────────────
    r_colors = [DANGER if v > ucl_r else WARN for v in ranges]
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(
        x=list(range(n_sg)), y=ranges,
        mode="lines+markers",
        line=dict(color=WARN, width=1.5),
        marker=dict(color=r_colors, size=7),
        name="Range",
    ))
    fig_r.add_hline(y=ucl_r, line=dict(color=DANGER, dash="dash"), annotation_text="UCL-R", annotation_font_color=DANGER)
    fig_r.add_hline(y=rmean, line=dict(color=OK,     dash="dot"),  annotation_text="R̄",    annotation_font_color=OK)
    fig_r.update_layout(
        title="R Chart (Subgroup Ranges)",
        xaxis_title="Subgroup", yaxis_title="Range",
        template=DARK, height=280,
        plot_bgcolor="#0d1526", paper_bgcolor="#080c14",
    )
    st.plotly_chart(fig_r, use_container_width=True)

    # ── CUSUM ─────────────────────────────────────────────────────────────────
    st.markdown("#### CUSUM Chart — Detects Subtle Sustained Process Shifts")
    cp, cn, h = cusum(data_arr)
    fig_cu = go.Figure()
    fig_cu.add_trace(go.Scatter(y=cp, name="CUSUM+", line=dict(color=ACCENT, width=2)))
    fig_cu.add_trace(go.Scatter(y=cn, name="CUSUM−", line=dict(color="#ff6b6b", width=2)))
    fig_cu.add_hline(y= h, line=dict(color=DANGER, dash="dash"), annotation_text=f"+h={h:.4f}", annotation_font_color=DANGER)
    fig_cu.add_hline(y=-h, line=dict(color=DANGER, dash="dash"), annotation_text=f"−h", annotation_font_color=DANGER)
    fig_cu.add_hline(y= 0, line=dict(color="gray",  dash="dot"))
    fig_cu.update_layout(
        title="CUSUM Control Chart",
        xaxis_title="Lot Index", yaxis_title="Cumulative Sum",
        template=DARK, height=300,
        plot_bgcolor="#0d1526", paper_bgcolor="#080c14",
    )
    st.plotly_chart(fig_cu, use_container_width=True)

    # ── Out-of-control table ──────────────────────────────────────────────────
    ooc = [(i, xbars[i]) for i in range(n_sg) if xbars[i] > ucl_x or xbars[i] < lcl_x]
    if ooc:
        st.markdown(
            f'<div class="alert-box">⚠ {len(ooc)} out-of-control subgroups detected! '
            f'Immediate investigation required.</div>',
            unsafe_allow_html=True,
        )
        ooc_df = pd.DataFrame(ooc, columns=["Subgroup", f"Mean {label}"])
        ooc_df["Direction"] = ooc_df[f"Mean {label}"].apply(
            lambda v: "🔴 Above UCL" if v > ucl_x else "🔴 Below LCL"
        )
        st.dataframe(ooc_df, use_container_width=True)
    else:
        st.markdown('<div class="ok-box">✅ All subgroups within control limits — Process in statistical control.</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — YIELD ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    st.subheader("Lot-by-Lot Yield Trend & Loss Breakdown")

    ca, cb = st.columns([3, 2])

    with ca:
        marker_colors = [DANGER if e else ACCENT for e in lot["is_excursion"]]
        fig_y = go.Figure()
        fig_y.add_trace(go.Scatter(
            x=lot["lot_id"], y=lot["mean_yield"],
            mode="lines+markers",
            line=dict(color=ACCENT, width=1.5),
            marker=dict(color=marker_colors, size=7),
            name="Yield (%)",
        ))
        fig_y.add_hline(y=85,                       line=dict(color=OK,   dash="dash"), annotation_text="Target 85%", annotation_font_color=OK)
        fig_y.add_hline(y=lot["mean_yield"].mean(), line=dict(color=WARN, dash="dot"),
                        annotation_text=f"Avg {lot['mean_yield'].mean():.1f}%", annotation_font_color=WARN)

        tick_step = max(1, len(lot) // 15)
        fig_y.update_layout(
            title="Lot Yield Trend (🔴 = Excursion Lots)",
            xaxis=dict(tickangle=45,
                       tickvals=lot["lot_id"].iloc[::tick_step].tolist(),
                       ticktext=lot["lot_id"].iloc[::tick_step].tolist()),
            yaxis_title="Yield (%)",
            template=DARK, height=380,
            plot_bgcolor="#0d1526", paper_bgcolor="#080c14",
        )
        st.plotly_chart(fig_y, use_container_width=True)

    with cb:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=lot["mean_yield"], nbinsx=25,
            marker_color=ACCENT, opacity=0.75, name="Yield dist."
        ))
        fig_hist.add_vline(x=85,                      line=dict(color=OK,   dash="dash"), annotation_text="Target", annotation_font_color=OK)
        fig_hist.add_vline(x=lot["mean_yield"].mean(), line=dict(color=WARN, dash="dot"),  annotation_text="Mean",   annotation_font_color=WARN)
        fig_hist.update_layout(
            title="Yield Distribution",
            xaxis_title="Yield (%)", yaxis_title="# Lots",
            template=DARK, height=380,
            plot_bgcolor="#0d1526", paper_bgcolor="#080c14",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    cc, cd = st.columns(2)

    with cc:
        fig_sc = px.scatter(
            lot, x="mean_d0", y="mean_yield",
            color="is_excursion",
            color_discrete_map={True: DANGER, False: ACCENT},
            trendline="ols",
            title="Defect Density D0 vs Yield (Murphy's Model)",
            labels={"mean_d0": "D0 (cm⁻²)", "mean_yield": "Yield (%)"},
        )
        fig_sc.update_layout(template=DARK, height=370,
                             plot_bgcolor="#0d1526", paper_bgcolor="#080c14")
        st.plotly_chart(fig_sc, use_container_width=True)

    with cd:
        step_y = df.groupby("process_step")["yield_pct"].mean().sort_values()
        bar_c  = [DANGER if v < 82 else WARN if v < 85 else ACCENT for v in step_y.values]
        fig_bar = go.Figure(go.Bar(
            x=step_y.values, y=step_y.index,
            orientation="h",
            marker_color=bar_c,
            text=[f"{v:.1f}%" for v in step_y.values],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="Avg Yield by Process Step",
            xaxis_title="Yield (%)",
            template=DARK, height=370,
            plot_bgcolor="#0d1526", paper_bgcolor="#080c14",
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ROOT CAUSE & PCA
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    st.subheader("Root Cause Analysis — PCA + Isolation Forest Anomaly Detection")

    FEATURES = ["mean_vt", "mean_idsat", "mean_rs", "mean_cr", "mean_d0"]
    FEAT_NAMES = ["Vt NMOS", "Idsat NMOS", "Sheet Res.", "Contact Res.", "Defect Density"]

    X = lot[FEATURES].values
    X_sc = StandardScaler().fit_transform(X)

    pca   = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_sc)
    var   = pca.explained_variance_ratio_ * 100

    iso       = IsolationForest(contamination=0.10, random_state=42)
    ano_label = iso.fit_predict(X_sc)
    is_ano    = ano_label == -1

    pa, pb = st.columns([3, 2])

    with pa:
        pca_df = pd.DataFrame({
            "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
            "Lot":      lot["lot_id"],
            "Yield":    lot["mean_yield"],
            "Status":   ["🔴 Anomaly" if a else "🔵 Normal" for a in is_ano],
        })
        fig_pca = px.scatter(
            pca_df, x="PC1", y="PC2",
            color="Status", size="Yield",
            hover_data=["Lot", "Yield"],
            color_discrete_map={"🔴 Anomaly": DANGER, "🔵 Normal": ACCENT},
            title=f"PCA — PC1 ({var[0]:.1f}%) vs PC2 ({var[1]:.1f}%)",
        )
        fig_pca.update_layout(template=DARK, height=430,
                              plot_bgcolor="#0d1526", paper_bgcolor="#080c14")
        st.plotly_chart(fig_pca, use_container_width=True)

    with pb:
        fig_ev = go.Figure(go.Bar(
            x=[f"PC{i+1}" for i in range(3)],
            y=var,
            marker_color=[ACCENT, WARN, "#ff6b6b"],
            text=[f"{v:.1f}%" for v in var],
            textposition="outside",
        ))
        fig_ev.update_layout(
            title="Explained Variance",
            yaxis_title="%",
            template=DARK, height=220,
            plot_bgcolor="#0d1526", paper_bgcolor="#080c14",
        )
        st.plotly_chart(fig_ev, use_container_width=True)

        st.metric("🔴 Anomalous Lots", int(is_ano.sum()))
        st.metric("✅ Normal Lots",    int((~is_ano).sum()))

        st.markdown("**PC1 Top Contributors:**")
        loadings = pd.Series(np.abs(pca.components_[0]), index=FEAT_NAMES).sort_values(ascending=False)
        for feat, val in loadings.items():
            bar = "█" * max(1, int(val * 18))
            st.markdown(
                f"<span style='font-family:monospace;font-size:12px;color:#5a8abf'>"
                f"{feat[:16]:16s}</span>"
                f"<span style='color:{ACCENT}'> {bar}</span>"
                f"<span style='color:#888'> {val:.3f}</span>",
                unsafe_allow_html=True,
            )

    # Correlation heatmap
    st.markdown("#### WAT Parameter Correlation Heatmap")
    corr = lot[FEATURES].corr()
    corr.index = FEAT_NAMES
    corr.columns = FEAT_NAMES
    fig_hm = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        title="Parameter Correlations (Lot-level Means)",
    )
    fig_hm.update_layout(template=DARK, height=380, paper_bgcolor="#080c14")
    st.plotly_chart(fig_hm, use_container_width=True)

    # Anomaly table
    st.markdown("#### 🔴 Flagged Anomalous Lots (Isolation Forest)")
    ano_lot = lot[is_ano][["lot_id", "mean_yield", "mean_vt", "mean_idsat", "mean_d0"]].copy()
    ano_lot.columns = ["Lot ID", "Yield (%)", "Vt NMOS (V)", "Idsat (µA/µm)", "D0 (cm⁻²)"]
    st.dataframe(ano_lot.round(4), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CPK & PARAMETER HEALTH
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    st.subheader("Process Capability (Cpk) & WAT Parameter Health")

    SPEC_TABLE = {
        "mean_vt":    ("Vt NMOS (V)",           0.40,  0.50),
        "mean_idsat": ("Idsat NMOS (µA/µm)",     430.0, 530.0),
        "mean_rs":    ("Sheet Resistance (Ω/□)", 85.0,  115.0),
        "mean_cr":    ("Contact Resistance (Ω)", 42.0,  58.0),
        "mean_d0":    ("Defect Density (cm⁻²)",  0.0,   0.50),
    }

    rows = []
    for col_k, (lbl, sl, su) in SPEC_TABLE.items():
        v   = lot[col_k].values
        cpk = compute_cpk(v, sl, su)
        status = "✅ Capable" if cpk >= 1.33 else "⚠️ Marginal" if cpk >= 1.0 else "❌ Not Capable"
        rows.append({
            "Parameter": lbl,
            "Mean":      round(float(v.mean()), 4),
            "Std Dev":   round(float(v.std(ddof=1)), 5),
            "LSL":       sl, "USL": su,
            "Cpk":       cpk, "Status": status,
        })

    cpk_df = pd.DataFrame(rows)
    st.dataframe(cpk_df, use_container_width=True, height=220)

    # Cpk bar chart
    bar_clr = [OK if r["Cpk"] >= 1.33 else WARN if r["Cpk"] >= 1.0 else DANGER for r in rows]
    fig_cpk = go.Figure(go.Bar(
        x=cpk_df["Parameter"], y=cpk_df["Cpk"],
        marker_color=bar_clr,
        text=cpk_df["Cpk"].astype(str),
        textposition="outside",
    ))
    fig_cpk.add_hline(y=1.33, line=dict(color=OK,   dash="dash", width=2),
                      annotation_text="Cpk = 1.33  (6σ target)",  annotation_font_color=OK)
    fig_cpk.add_hline(y=1.0,  line=dict(color=WARN, dash="dash", width=1.5),
                      annotation_text="Cpk = 1.00  (minimum)",    annotation_font_color=WARN)
    fig_cpk.update_layout(
        title="Process Capability Index (Cpk) per Parameter",
        yaxis_title="Cpk",
        template=DARK, height=380,
        plot_bgcolor="#0d1526", paper_bgcolor="#080c14",
    )
    st.plotly_chart(fig_cpk, use_container_width=True)

    # Histograms vs spec limits (first 3 params)
    st.markdown("#### Parameter Distributions vs Spec Limits")
    h_cols = st.columns(3)
    for idx, (col_k, (lbl, sl, su)) in enumerate(list(SPEC_TABLE.items())[:3]):
        with h_cols[idx]:
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(
                x=lot[col_k], nbinsx=20,
                marker_color=ACCENT, opacity=0.75,
            ))
            fig_h.add_vline(x=sl, line=dict(color=DANGER, dash="dash"), annotation_text="LSL", annotation_font_color=DANGER)
            fig_h.add_vline(x=su, line=dict(color=DANGER, dash="dash"), annotation_text="USL", annotation_font_color=DANGER)
            fig_h.add_vline(x=lot[col_k].mean(), line=dict(color="yellow", dash="dot"), annotation_text="μ", annotation_font_color="yellow")
            fig_h.update_layout(
                title=lbl[:28], template=DARK, height=260, showlegend=False,
                plot_bgcolor="#0d1526", paper_bgcolor="#080c14",
            )
            st.plotly_chart(fig_h, use_container_width=True)

    # Radar chart
    st.markdown("#### Process Health Radar")
    cats    = [r["Parameter"] for r in rows]
    norm    = [min(r["Cpk"] / 1.67, 1.0) for r in rows]
    tgt_val = 1.33 / 1.67

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=norm + [norm[0]], theta=cats + [cats[0]],
        fill="toself",
        fillcolor="rgba(0,204,255,0.12)",
        line=dict(color=ACCENT, width=2),
        name="Cpk (norm)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[tgt_val] * len(cats) + [tgt_val],
        theta=cats + [cats[0]],
        mode="lines",
        line=dict(color=OK, dash="dash", width=1.5),
        name="Target (1.33)",
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template=DARK, height=430,
        title="Process Health Radar (Cpk normalised to 1.67)",
        paper_bgcolor="#080c14",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.divider()
st.caption("🔬 Semiconductor Yield & SPC Dashboard · Personal Project")
