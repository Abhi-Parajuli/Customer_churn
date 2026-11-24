"""Churn Radar - Telecom customer churn prediction and retention targeting (Streamlit)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Churn Radar", page_icon="📡", layout="wide")

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "telco_churn.csv"
MODEL_PATH = BASE / "models" / "model_v1.joblib"
FEATS_PATH = BASE / "models" / "feature_columns_v1.joblib"
META_PATH = BASE / "models" / "model_meta_v1.json"
OOF_PATH = BASE / "models" / "oof_predictions_v1.csv"

# ----------------------------------------------------------------------------
# Theme ("Signal": warm paper background, navy ink, coral = churn, teal = stay)
# ----------------------------------------------------------------------------
INK, PAPER, CARD = "#17233B", "#F5F1E8", "#FFFFFF"
CORAL, TEAL, GOLD, MUTED, LINE = "#E4572E", "#177E89", "#F0A202", "#6B7280", "#E3DDCE"
STATUS_COLORS = {"Stay": TEAL, "Churn": CORAL}

st.markdown(
    f"""
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stMarkdown, .stSelectbox, .stSlider, .stNumberInput {{ font-family: 'Manrope', sans-serif; }}
.stApp {{ background: {PAPER}; color: {INK}; }}
.block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
h1, h2, h3, h4 {{ font-family: 'Fraunces', serif !important; color: {INK}; letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid {LINE}; }}
.hero {{ background: {INK}; color: {PAPER}; border-radius: 22px; padding: 30px 36px; margin-bottom: 22px;
        position: relative; overflow: hidden; }}
.hero:after {{ content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px; border-radius: 50%;
        border: 2px solid rgba(240,162,2,.35); box-shadow: 0 0 0 38px rgba(240,162,2,.08), 0 0 0 76px rgba(240,162,2,.05); }}
.hero .eyebrow {{ font-size: .78rem; letter-spacing: .16em; text-transform: uppercase; color: {GOLD}; font-weight: 700; }}
.hero h1 {{ color: {PAPER} !important; font-size: 2.5rem; margin: .25rem 0 .4rem 0; }}
.hero p {{ color: #CFD6E4; max-width: 720px; margin: 0; font-size: 1rem; line-height: 1.55; }}
.kpi {{ background: {CARD}; border: 1px solid {LINE}; border-left: 5px solid {INK}; border-radius: 14px; padding: 14px 18px; height: 100%; min-height: 118px; }}
.kpi.coral {{ border-left-color: {CORAL}; }} .kpi.teal {{ border-left-color: {TEAL}; }} .kpi.gold {{ border-left-color: {GOLD}; }}
.kpi-l {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .09em; color: {MUTED}; font-weight: 700; }}
.kpi-v {{ font-family: 'Fraunces', serif; font-size: 1.85rem; font-weight: 700; color: {INK}; line-height: 1.15; }}
.kpi-s {{ font-size: .8rem; color: {MUTED}; margin-top: 2px; }}
.verdict {{ border-radius: 16px; padding: 16px 20px; font-weight: 600; margin: 4px 0 10px 0; }}
.verdict.act {{ background: #FCE6DF; border: 1px solid {CORAL}; color: #8F2A0E; }}
.verdict.hold {{ background: #DDF0F1; border: 1px solid {TEAL}; color: #0E4F57; }}
.verdict small {{ display:block; font-weight: 500; margin-top: 3px; }}
.section-note {{ color: {MUTED}; font-size: .9rem; margin: -6px 0 10px 0; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 700; padding: 10px 18px; border-radius: 10px 10px 0 0; }}
.stTabs [aria-selected="true"] {{ background: {CARD}; color: {CORAL}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {CORAL}; }}
</style>
""",
    unsafe_allow_html=True,
)


def style(fig, height=340, legend=True):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=44, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color=INK, size=12),
        title=dict(font=dict(family="Fraunces, serif", size=17), x=0.01),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right", title_text=""),
        showlegend=legend,
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zeroline=False, linecolor=LINE)
    return fig


def kpi(label, value, sub="", tone=""):
    return (f'<div class="kpi {tone}"><div class="kpi-l">{label}</div>'
            f'<div class="kpi-v">{value}</div><div class="kpi-s">{sub}</div></div>')


# ----------------------------------------------------------------------------
# Loading (same cleaning rules as the notebook)
# ----------------------------------------------------------------------------
YES_NO = ["Partner", "Dependents", "PhoneService", "MultipleLines", "PaperlessBilling", "OnlineSecurity",
          "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
PROTECTION = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(FEATS_PATH)


@st.cache_data
def load_meta():
    with open(META_PATH) as f:
        return json.load(f)


@st.cache_data
def load_oof():
    return pd.read_csv(OOF_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df[df["tenure"] > 0].drop(columns="customerID").reset_index(drop=True)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    for c in YES_NO:
        df[c] = df[c].replace({"No phone service": "No", "No internet service": "No"}).map({"Yes": 1, "No": 0})
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["Status"] = df["Churn"].map({0: "Stay", 1: "Churn"})
    df["Tenure band"] = pd.cut(df["tenure"], [0, 12, 24, 48, 72], labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"])
    df["Charge band"] = pd.cut(df["MonthlyCharges"], [0, 30, 50, 70, 90, 125],
                               labels=["<$30", "$30-50", "$50-70", "$70-90", "$90+"])
    return df


try:
    model, FEATS = load_model()
    meta = load_meta()
    oof = load_oof()
    df = load_data()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Run the notebook's final cell to create the files in `models/`.")
    st.stop()

THRESHOLDS = np.arange(0.05, 0.96, 0.01)


@st.cache_data
def cost_curve(offer, loss, save):
    y, p = oof["y"].to_numpy(), oof["p"].to_numpy()
    costs = []
    for t in THRESHOLDS:
        flag = p >= t
        costs.append(flag.sum() * offer + ((~flag) & (y == 1)).sum() * loss + (flag & (y == 1)).sum() * loss * (1 - save))
    return np.array(costs), y.sum() * loss


# ----------------------------------------------------------------------------
# Sidebar: business assumptions shared by every tab
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Retention economics")
    st.caption("Assumptions, not measured values. The decision threshold is re-tuned from them live.")
    base_c = meta["costs"]
    offer = st.slider("Cost of a retention offer ($)", 10, 200, int(base_c["offer"]), 5)
    loss = st.slider("Value lost per churner ($)", 100, 1500, int(base_c["loss"]), 50)
    save = st.slider("Offer success rate", 0.10, 0.90, float(base_c["save_rate"]), 0.05)
    curve, no_action_cost = cost_curve(offer, loss, save)
    live_thr = float(THRESHOLDS[curve.argmin()])
    live_saving = 1 - curve.min() / no_action_cost
    st.divider()
    st.metric("Flag customers above", f"{live_thr:.0%} risk")
    st.metric("Expected saving vs. no action", f"{live_saving:.1%}")
    st.caption(f"Break-even risk: {offer / (save * loss):.0%}  (offer cost / (success rate x loss))")

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown(
    f"""<div class="hero"><div class="eyebrow">Telecom · Retention analytics</div>
<h1>Churn Radar</h1>
<p>Predict which customers are likely to leave, explore what drives churn across {len(df):,} subscribers,
and choose who gets a retention offer using a cost-based threshold instead of a default 0.5.</p></div>""",
    unsafe_allow_html=True,
)

tab_predict, tab_dash, tab_model = st.tabs(["Predict", "Dashboard", "Model & cost"])

# ============================================================================
# PREDICT
# ============================================================================
with tab_predict:
    left, right = st.columns([1.25, 1], gap="large")

    with left:
        st.markdown("#### Customer profile")
        a, b = st.columns(2)
        with a:
            st.markdown("**Account**")
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                      "Bank transfer (automatic)", "Credit card (automatic)"])
            paperless = st.toggle("Paperless billing", value=True)
        with b:
            st.markdown("**Billing**")
            monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0, 0.5)
            manual_total = st.toggle("Enter total charges manually", value=False)
            total = st.number_input("Total charges ($)", 0.0, 9000.0, round(tenure * monthly, 2), 10.0) \
                if manual_total else round(tenure * monthly, 2)
            if not manual_total:
                st.caption(f"Total charges = tenure x monthly = ${total:,.2f}")
            st.markdown("**Household**")
            g1, g2 = st.columns(2)
            gender = g1.selectbox("Gender", ["Female", "Male"])
            senior = g2.selectbox("Senior citizen", ["No", "Yes"])
            partner = g1.toggle("Partner", value=False)
            dependents = g2.toggle("Dependents", value=False)

        st.markdown("**Services**")
        s1, s2, s3 = st.columns(3)
        internet = s1.selectbox("Internet", ["Fiber optic", "DSL", "No"])
        phone = s2.toggle("Phone service", value=True)
        multi = s3.toggle("Multiple lines", value=False, disabled=not phone)
        no_net = internet == "No"
        t1, t2, t3 = st.columns(3)
        security = t1.toggle("Online security", value=False, disabled=no_net)
        backup = t2.toggle("Online backup", value=False, disabled=no_net)
        device = t3.toggle("Device protection", value=False, disabled=no_net)
        t4, t5, t6 = st.columns(3)
        techsup = t4.toggle("Tech support", value=False, disabled=no_net)
        stv = t5.toggle("Streaming TV", value=False, disabled=no_net)
        smov = t6.toggle("Streaming movies", value=False, disabled=no_net)

    def build_row(**over):
        row = {
            "gender": gender, "SeniorCitizen": senior, "Partner": int(partner), "Dependents": int(dependents),
            "tenure": tenure, "PhoneService": int(phone), "MultipleLines": int(multi and phone),
            "InternetService": internet,
            "OnlineSecurity": int(security and not no_net), "OnlineBackup": int(backup and not no_net),
            "DeviceProtection": int(device and not no_net), "TechSupport": int(techsup and not no_net),
            "StreamingTV": int(stv and not no_net), "StreamingMovies": int(smov and not no_net),
            "Contract": contract, "PaperlessBilling": int(paperless), "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }
        row.update(over)
        return row

    def churn_prob(row):
        return float(model.predict_proba(pd.DataFrame([row])[FEATS])[:, 1][0])

    row = build_row()
    p = churn_prob(row)
    ev = p * save * loss - offer          # expected net value of making the offer

    with right:
        st.markdown("#### Churn risk")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number=dict(suffix="%", font=dict(family="Fraunces, serif", size=54, color=INK)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=MUTED),
                bar=dict(color=INK, thickness=0.28), bgcolor=CARD, borderwidth=0,
                steps=[dict(range=[0, live_thr * 100], color="#BFE3E5"),
                       dict(range=[live_thr * 100, 60], color="#FBDC9B"),
                       dict(range=[60, 100], color="#F5B7A3")],
                threshold=dict(line=dict(color=CORAL, width=4), thickness=0.85, value=live_thr * 100),
            )))
        st.plotly_chart(style(fig, 260, legend=False), width="stretch")

        if p >= live_thr:
            st.markdown(f'<div class="verdict act">Make a retention offer<small>Risk is above the {live_thr:.0%} '
                        f'cost-based threshold.</small></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict hold">No offer needed<small>Risk is below the {live_thr:.0%} '
                        f'cost-based threshold.</small></div>', unsafe_allow_html=True)

        seg = df[(df["Contract"] == contract) & (df["InternetService"] == internet)]
        k1, k2 = st.columns(2)
        k1.markdown(kpi("Expected value of an offer", f"{'+' if ev >= 0 else '-'}${abs(ev):,.0f}",
                        "risk x success x loss - offer cost", "teal" if ev >= 0 else "coral"), unsafe_allow_html=True)
        k2.markdown(kpi("Similar customers churn", f"{seg['Churn'].mean():.0%}",
                        f"{contract} · {internet} (n={len(seg):,})", "gold"), unsafe_allow_html=True)

    # What-if levers --------------------------------------------------------
    st.markdown("#### What would move this customer's risk?")
    st.markdown('<div class="section-note">Each bar re-scores the same customer with one change. '
                'Associations from historical data, not guaranteed effects.</div>', unsafe_allow_html=True)
    scenarios = {}
    if contract != "One year":
        scenarios["Move to a 1-year contract"] = dict(Contract="One year")
    if contract != "Two year":
        scenarios["Move to a 2-year contract"] = dict(Contract="Two year")
    if payment in ("Electronic check", "Mailed check"):
        scenarios["Switch to automatic card payment"] = dict(PaymentMethod="Credit card (automatic)")
    if not no_net and not row["TechSupport"]:
        scenarios["Add tech support"] = dict(TechSupport=1)
    if not no_net and not row["OnlineSecurity"]:
        scenarios["Add online security"] = dict(OnlineSecurity=1)
    if monthly > 25:
        scenarios["Reduce monthly bill by $10"] = dict(MonthlyCharges=monthly - 10,
                                                       TotalCharges=total if manual_total else round(tenure * (monthly - 10), 2))
    if scenarios:
        res = pd.DataFrame([(name, churn_prob({**row, **chg}) - p) for name, chg in scenarios.items()],
                           columns=["Change", "delta"]).sort_values("delta")
        fig = go.Figure(go.Bar(
            x=res["delta"] * 100, y=res["Change"], orientation="h",
            marker_color=[TEAL if d < 0 else CORAL for d in res["delta"]],
            text=[f"{d * 100:+.1f} pts" for d in res["delta"]], textposition="outside", cliponaxis=False))
        fig.update_xaxes(title="Change in churn probability (percentage points)")
        fig.update_layout(title="Scenario impact on churn risk")
        st.plotly_chart(style(fig, 300, legend=False), width="stretch")
    else:
        st.info("This customer is already on the lowest-risk option for every lever shown.")

# ============================================================================
# DASHBOARD
# ============================================================================
with tab_dash:
    f1, f2 = st.columns(2)
    sel_c = f1.multiselect("Contract", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique()))
    sel_i = f2.multiselect("Internet service", sorted(df["InternetService"].unique()),
                           default=sorted(df["InternetService"].unique()))
    dff = df[df["Contract"].isin(sel_c) & df["InternetService"].isin(sel_i)]

    if dff.empty:
        st.warning("No customers match the selected filters.")
    else:
        overall = dff["Churn"].mean()
        lost_rev = dff.loc[dff["Churn"] == 1, "MonthlyCharges"].sum()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(kpi("Customers", f"{len(dff):,}", "billed subscribers"), unsafe_allow_html=True)
        c2.markdown(kpi("Churn rate", f"{overall:.1%}", f"{int(dff['Churn'].sum()):,} churned", "coral"), unsafe_allow_html=True)
        c3.markdown(kpi("Avg monthly charges", f"${dff['MonthlyCharges'].mean():,.2f}",
                        f"churners ${dff.loc[dff.Churn == 1, 'MonthlyCharges'].mean():,.0f} · stayers "
                        f"${dff.loc[dff.Churn == 0, 'MonthlyCharges'].mean():,.0f}", "gold"), unsafe_allow_html=True)
        c4.markdown(kpi("Monthly revenue lost", f"${lost_rev:,.0f}", "sum of churners' monthly charges", "coral"),
                    unsafe_allow_html=True)
        c5.markdown(kpi("Avg tenure", f"{dff['tenure'].mean():.1f} mo", "months with the company", "teal"),
                    unsafe_allow_html=True)
        st.write("")

        def churn_bar(col, title):
            t = dff.groupby(col, observed=True)["Churn"].agg(["mean", "size"]).reset_index()
            fig = go.Figure(go.Bar(x=t[col].astype(str), y=t["mean"] * 100, marker_color=CORAL,
                                   text=[f"{v:.0%}" for v in t["mean"]], textposition="outside",
                                   customdata=t["size"], hovertemplate="%{x}<br>%{y:.1f}% churn<br>n=%{customdata:,}<extra></extra>"))
            fig.add_hline(y=overall * 100, line_dash="dash", line_color=INK, annotation_text="average",
                          annotation_position="top left")
            fig.update_yaxes(title="churn %", range=[0, max(t["mean"].max() * 100 * 1.2, 5)])
            fig.update_layout(title=title)
            return style(fig, 320, legend=False)

        r1a, r1b = st.columns(2)
        r1a.plotly_chart(churn_bar("Contract", "Churn rate by contract"), width="stretch")
        r1b.plotly_chart(churn_bar("InternetService", "Churn rate by internet service"), width="stretch")

        st.markdown("#### Monthly charges")
        m1, m2 = st.columns(2)
        fig = px.histogram(dff, x="MonthlyCharges", color="Status", nbins=40, barmode="overlay", opacity=0.7,
                           marginal="box", color_discrete_map=STATUS_COLORS,
                           title="Monthly charges: churners vs. stayers")
        fig.update_xaxes(title="monthly charges ($)")
        m1.plotly_chart(style(fig, 360), width="stretch")
        m2.plotly_chart(churn_bar("Charge band", "Churn rate by monthly-charge band"), width="stretch")

        m3, m4 = st.columns(2)
        sample = dff.sample(min(len(dff), 1800), random_state=42)
        fig = px.scatter(sample, x="tenure", y="MonthlyCharges", color="Status", opacity=0.55,
                         color_discrete_map=STATUS_COLORS, title="Monthly charges vs. tenure")
        fig.update_xaxes(title="tenure (months)"); fig.update_yaxes(title="monthly charges ($)")
        m3.plotly_chart(style(fig, 360), width="stretch")
        avg = dff.groupby(["Contract", "Status"], observed=True)["MonthlyCharges"].mean().reset_index()
        fig = px.bar(avg, x="Contract", y="MonthlyCharges", color="Status", barmode="group",
                     color_discrete_map=STATUS_COLORS, text_auto=".0f", title="Average monthly charges by contract")
        fig.update_yaxes(title="avg monthly charges ($)")
        m4.plotly_chart(style(fig, 360), width="stretch")

        st.markdown("#### Tenure, payment and services")
        t1, t2 = st.columns(2)
        fig = px.histogram(dff, x="tenure", color="Status", nbins=36, barmode="overlay", opacity=0.7,
                           color_discrete_map=STATUS_COLORS, title="Tenure distribution")
        fig.update_xaxes(title="tenure (months)")
        t1.plotly_chart(style(fig, 340), width="stretch")
        t2.plotly_chart(churn_bar("Tenure band", "Churn rate by tenure band"), width="stretch")

        h1, h2 = st.columns(2)
        piv = dff.pivot_table(index="Contract", columns="InternetService", values="Churn", aggfunc="mean") * 100
        fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index,
                                   colorscale=[[0, "#FBF7EE"], [0.5, "#F5B7A3"], [1, CORAL]],
                                   text=[[f"{v:.0f}%" if pd.notna(v) else "" for v in r] for r in piv.values],
                                   texttemplate="%{text}", colorbar=dict(title="churn %")))
        fig.update_layout(title="Churn rate: contract x internet service")
        h1.plotly_chart(style(fig, 340, legend=False), width="stretch")
        h2.plotly_chart(churn_bar("PaymentMethod", "Churn rate by payment method"), width="stretch")

        net = dff[dff["InternetService"] != "No"].copy()
        if not net.empty:
            net["Protection add-ons"] = net[PROTECTION].sum(axis=1)
            t = net.groupby("Protection add-ons")["Churn"].agg(["mean", "size"]).reset_index()
            fig = go.Figure(go.Bar(x=t["Protection add-ons"].astype(str), y=t["mean"] * 100, marker_color=TEAL,
                                   text=[f"{v:.0%}" for v in t["mean"]], textposition="outside",
                                   customdata=t["size"], hovertemplate="%{x} add-ons<br>%{y:.1f}% churn<br>n=%{customdata:,}<extra></extra>"))
            fig.update_yaxes(title="churn %", range=[0, t["mean"].max() * 100 * 1.2])
            fig.update_xaxes(title="number of protection add-ons (security, backup, device, tech support)")
            fig.update_layout(title="Churn rate by protection add-ons (internet customers)")
            st.plotly_chart(style(fig, 320, legend=False), width="stretch")

# ============================================================================
# MODEL & COST
# ============================================================================
with tab_model:
    T = meta["test"]
    ci = T["ci"]
    st.markdown(f"#### {meta['model_name']} · held-out test set ({meta['n_test']:,} customers)")
    k = st.columns(5)
    k[0].markdown(kpi("ROC-AUC", f"{T['roc_auc']:.3f}", f"95% CI {ci['roc_auc'][0]:.3f}-{ci['roc_auc'][1]:.3f}", "teal"), unsafe_allow_html=True)
    k[1].markdown(kpi("PR-AUC", f"{T['pr_auc']:.3f}", f"95% CI {ci['pr_auc'][0]:.3f}-{ci['pr_auc'][1]:.3f}", "teal"), unsafe_allow_html=True)
    k[2].markdown(kpi("Churn recall", f"{T['recall']:.0%}", f"at the {meta['threshold']:.2f} threshold", "gold"), unsafe_allow_html=True)
    k[3].markdown(kpi("Precision", f"{T['precision']:.0%}", "of flagged customers churn", "gold"), unsafe_allow_html=True)
    k[4].markdown(kpi("Cost saving", f"{T['saving_vs_no_action']:.1%}",
                      f"vs. no action · CI {ci['saving'][0]:.1%}-{ci['saving'][1]:.1%} (assumed costs)", "coral"), unsafe_allow_html=True)
    st.write("")

    a, b = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=THRESHOLDS, y=curve, mode="lines", line=dict(color=CORAL, width=3), name="Total cost"))
    fig.add_hline(y=no_action_cost, line_dash="dash", line_color=MUTED, annotation_text="flag nobody")
    fig.add_vline(x=live_thr, line_color=INK, annotation_text=f"best: {live_thr:.2f}")
    fig.add_vline(x=offer / (save * loss), line_dash="dot", line_color=TEAL, annotation_text="break-even", annotation_position="bottom right")
    fig.update_xaxes(title="decision threshold"); fig.update_yaxes(title="total cost, training OOF ($)")
    fig.update_layout(title="Cost vs. threshold (move the sidebar sliders)")
    a.plotly_chart(style(fig, 360, legend=False), width="stretch")

    cm = T["confusion"]
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["Pred: stay", "Pred: churn"], y=["Actual: stay", "Actual: churn"],
                               colorscale=[[0, "#FBF7EE"], [1, TEAL]], text=z, texttemplate="%{text}",
                               textfont=dict(size=20), showscale=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=f"Confusion matrix at threshold {meta['threshold']:.2f}")
    b.plotly_chart(style(fig, 360, legend=False), width="stretch")

    c, d = st.columns(2)
    cv = pd.DataFrame(meta["cv_comparison"]).sort_values("pr_auc")
    fig = go.Figure(go.Bar(x=cv["pr_auc"], y=cv["model"], orientation="h", marker_color=GOLD,
                           text=[f"{v:.3f}" for v in cv["pr_auc"]], textposition="outside", cliponaxis=False))
    fig.update_xaxes(title="5-fold CV PR-AUC (training split)", range=[0, cv["pr_auc"].max() * 1.15])
    fig.update_layout(title="Model comparison")
    c.plotly_chart(style(fig, 360, legend=False), width="stretch")

    imp = pd.DataFrame(meta["importance"]).sort_values("drop_in_pr_auc")
    fig = go.Figure(go.Bar(x=imp["drop_in_pr_auc"], y=imp["feature"], orientation="h", marker_color=INK,
                           error_x=dict(array=imp["sd"], color=MUTED)))
    fig.update_xaxes(title="drop in test PR-AUC when the feature is shuffled")
    fig.update_layout(title="What drives churn (permutation importance)")
    d.plotly_chart(style(fig, 360, legend=False), width="stretch")

    cal = meta["calibration"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dash"), name="perfect"))
    fig.add_trace(go.Scatter(x=cal["pred"], y=cal["obs"], mode="lines+markers", line=dict(color=CORAL, width=3), name="model"))
    fig.update_xaxes(title="predicted probability"); fig.update_yaxes(title="observed churn rate")
    fig.update_layout(title="Calibration (test set, deciles)")
    st.plotly_chart(style(fig, 340), width="stretch")

    st.caption("Cost figures are assumptions, so the dollar saving is illustrative of the method, not a forecast. "
               "Single data snapshot: no out-of-time validation. Drivers are associations, not causal effects.")
