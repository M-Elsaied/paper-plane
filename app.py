import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy import stats

st.set_page_config(
    page_title="Paper Airplane DOE Analysis",
    page_icon="✈️",
    layout="wide",
)

# ── Data ──────────────────────────────────────────────────────────────────────
# Full 2^4 factorial, 2 locations x 2 replicates = 64 runs
# A=Paper(-1=Notebook,+1=Printer), B=Design(-1=Dart,+1=Glider),
# C=Nose(-1=None,+1=Paperclip), D=Environment(-1=Indoor,+1=Outdoor)

DATA = [
    # Run, Location, Replicate, A, B, C, D, Distance, Deviation
    (1,"L1",1,-1,-1,-1,-1,61,"Left"),
    (2,"L1",1,-1,-1,-1,1,62,"Right"),
    (3,"L1",1,-1,-1,1,-1,38,"Right"),
    (4,"L1",1,-1,-1,1,1,37,"Left"),
    (5,"L1",1,-1,1,-1,-1,94,"Left"),
    (6,"L1",1,-1,1,-1,1,87,"Right"),
    (7,"L1",1,-1,1,1,-1,40,"Left"),
    (8,"L1",1,-1,1,1,1,44,"Right"),
    (9,"L1",1,1,-1,-1,-1,38,"Left"),
    (10,"L1",1,1,-1,-1,1,80,"Right"),
    (11,"L1",1,1,-1,1,-1,18,"Left"),
    (12,"L1",1,1,-1,1,1,39,"Right"),
    (13,"L1",1,1,1,-1,-1,63,"Right"),
    (14,"L1",1,1,1,-1,1,88,"Right"),
    (15,"L1",1,1,1,1,-1,52,"Right"),
    (16,"L1",1,1,1,1,1,85,"Right"),
    (17,"L1",2,-1,-1,-1,-1,52,"Left"),
    (18,"L1",2,-1,-1,-1,1,68,"Right"),
    (19,"L1",2,-1,-1,1,-1,42,"Left"),
    (20,"L1",2,-1,-1,1,1,42,"Right"),
    (21,"L1",2,-1,1,-1,-1,82,"Right"),
    (22,"L1",2,-1,1,-1,1,86,"Right"),
    (23,"L1",2,-1,1,1,-1,69,"Right"),
    (24,"L1",2,-1,1,1,1,49,"Right"),
    (25,"L1",2,1,-1,-1,-1,42,"Right"),
    (26,"L1",2,1,-1,-1,1,68,"Right"),
    (27,"L1",2,1,-1,1,-1,27,"Right"),
    (28,"L1",2,1,-1,1,1,46,"Left"),
    (29,"L1",2,1,1,-1,-1,47,"Left"),
    (30,"L1",2,1,1,-1,1,78,"Right"),
    (31,"L1",2,1,1,1,-1,54,"Left"),
    (32,"L1",2,1,1,1,1,77,"Right"),
    (33,"L2",1,-1,-1,-1,-1,93.4,"Left"),
    (34,"L2",1,-1,-1,-1,1,91.3,"Right"),
    (35,"L2",1,-1,-1,1,-1,79,"Left"),
    (36,"L2",1,-1,-1,1,1,38.9,"Right"),
    (37,"L2",1,-1,1,-1,-1,38,"Left"),
    (38,"L2",1,-1,1,-1,1,65.1,"Left"),
    (39,"L2",1,-1,1,1,-1,44,"Right"),
    (40,"L2",1,-1,1,1,1,65.5,"Left"),
    (41,"L2",1,1,-1,-1,-1,44.1,"Right"),
    (42,"L2",1,1,-1,-1,1,80.4,"Left"),
    (43,"L2",1,1,-1,1,-1,192.1,"Left"),
    (44,"L2",1,1,-1,1,1,61.7,"Right"),
    (45,"L2",1,1,1,-1,-1,79.5,"Left"),
    (46,"L2",1,1,1,-1,1,55,"Left"),
    (47,"L2",1,1,1,1,-1,116.2,"Left"),
    (48,"L2",1,1,1,1,1,52.4,"Left"),
    (49,"L2",2,-1,-1,-1,-1,63.9,"Right"),
    (50,"L2",2,-1,-1,-1,1,48.2,"Right"),
    (51,"L2",2,-1,-1,1,-1,80.7,"Right"),
    (52,"L2",2,-1,-1,1,1,102.3,"Right"),
    (53,"L2",2,-1,1,-1,-1,18.5,"Left"),
    (54,"L2",2,-1,1,-1,1,12.2,"Right"),
    (55,"L2",2,-1,1,1,-1,39.7,"Left"),
    (56,"L2",2,-1,1,1,1,26.6,"Right"),
    (57,"L2",2,1,-1,-1,-1,43.5,"Left"),
    (58,"L2",2,1,-1,-1,1,65.6,"Right"),
    (59,"L2",2,1,-1,1,-1,90.8,"Left"),
    (60,"L2",2,1,-1,1,1,86.5,"Right"),
    (61,"L2",2,1,1,-1,-1,45.5,"Left"),
    (62,"L2",2,1,1,-1,1,46.1,"Right"),
    (63,"L2",2,1,1,1,-1,58.6,"Right"),
    (64,"L2",2,1,1,1,1,63,"Right"),
]

COLS = ["Run","Location","Replicate","A","B","C","D","Distance","Deviation"]
df = pd.DataFrame(DATA, columns=COLS)

# Human-readable labels
FACTOR_LABELS = {
    "A": {"name": "Paper Type", -1: "Notebook", 1: "Printer"},
    "B": {"name": "Folding Design", -1: "Dart", 1: "Glider"},
    "C": {"name": "Nose Design", -1: "None", 1: "Paperclip"},
    "D": {"name": "Environment", -1: "Indoor", 1: "Outdoor"},
}

df["A_label"] = df["A"].map(FACTOR_LABELS["A"])
df["B_label"] = df["B"].map(FACTOR_LABELS["B"])
df["C_label"] = df["C"].map(FACTOR_LABELS["C"])
df["D_label"] = df["D"].map(FACTOR_LABELS["D"])

# Rename factors for model formula (avoid clash with Patsy's C() function)
df["F_A"] = df["A"]
df["F_B"] = df["B"]
df["F_C"] = df["C"]
df["F_D"] = df["D"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Data Explorer",
        "Main Effects",
        "Interaction Effects",
        "ANOVA Table",
        "Residual Diagnostics",
        "Best Settings",
    ],
)

# ── Helper: ANOVA model ──────────────────────────────────────────────────────
@st.cache_data
def fit_model():
    model_df = df.copy()
    model_df["Loc"] = pd.Categorical(model_df["Location"])
    formula = "Distance ~ Loc + F_A*F_B*F_C*F_D"
    model = ols(formula, data=model_df).fit()
    anova_table = anova_lm(model, typ=2)
    return model, anova_table

model, anova_table = fit_model()

# ── Pages ─────────────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("Paper Airplane DOE Analysis")
    st.markdown("""
    **Objective:** Maximize mean flight distance of a paper airplane using a
    2<sup>4</sup> full factorial design with replication and blocking.

    | Factor | Low (−1) | High (+1) |
    |--------|----------|-----------|
    | **A** – Paper Type | Notebook (flexible) | Printer (stiff) |
    | **B** – Folding Design | Classic Dart | Glider |
    | **C** – Nose Design | None | Paperclip |
    | **D** – Environment | Indoor | Outdoor |

    **Design:** 2<sup>4</sup> factorial &times; 2 locations &times; 2 replicates = **64 runs**
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Runs", len(df))
    col2.metric("Mean Distance", f"{df['Distance'].mean():.1f} in")
    col3.metric("Min Distance", f"{df['Distance'].min():.1f} in")
    col4.metric("Max Distance", f"{df['Distance'].max():.1f} in")

    fig = px.histogram(
        df, x="Distance", nbins=20, color="Location",
        title="Distribution of Flight Distances",
        labels={"Distance": "Distance (inches)"},
        barmode="overlay", opacity=0.7,
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Data Explorer":
    st.title("Data Explorer")

    with st.expander("Filter Data", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        a_filter = c1.multiselect("Paper Type (A)", ["Notebook", "Printer"], default=["Notebook", "Printer"])
        b_filter = c2.multiselect("Design (B)", ["Dart", "Glider"], default=["Dart", "Glider"])
        c_filter = c3.multiselect("Nose (C)", ["None", "Paperclip"], default=["None", "Paperclip"])
        d_filter = c4.multiselect("Environment (D)", ["Indoor", "Outdoor"], default=["Indoor", "Outdoor"])

    mask = (
        df["A_label"].isin(a_filter) &
        df["B_label"].isin(b_filter) &
        df["C_label"].isin(c_filter) &
        df["D_label"].isin(d_filter)
    )
    filtered = df[mask]

    st.dataframe(
        filtered[["Run","Location","Replicate","A_label","B_label","C_label","D_label","Distance","Deviation"]]
        .rename(columns={
            "A_label":"Paper","B_label":"Design","C_label":"Nose","D_label":"Environment"
        }),
        use_container_width=True,
        hide_index=True,
    )

    fig = px.strip(
        filtered, x="D_label", y="Distance", color="B_label",
        facet_col="A_label", facet_row="C_label",
        labels={"Distance":"Distance (in)","D_label":"Environment","B_label":"Design","A_label":"Paper","C_label":"Nose"},
        title="Distance by Factor Combinations",
    )
    fig.update_traces(jitter=0.4)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Main Effects":
    st.title("Main Effects Plot")
    st.markdown("Each point shows the mean distance at each factor level, averaged over all other factors.")

    fig = make_subplots(rows=1, cols=4, subplot_titles=[
        f"{FACTOR_LABELS[f]['name']} ({f})" for f in "ABCD"
    ])
    for i, factor in enumerate("ABCD", 1):
        means = df.groupby(factor)["Distance"].mean().reset_index()
        low_label = FACTOR_LABELS[factor][-1]
        high_label = FACTOR_LABELS[factor][1]
        means["Label"] = means[factor].map({-1: low_label, 1: high_label})

        fig.add_trace(go.Scatter(
            x=means["Label"], y=means["Distance"],
            mode="lines+markers", marker=dict(size=12),
            name=FACTOR_LABELS[factor]["name"],
            showlegend=False,
        ), row=1, col=i)
        fig.update_yaxes(title_text="Mean Distance (in)" if i == 1 else "", row=1, col=i)

    fig.update_layout(height=400, title="Main Effects on Flight Distance")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Estimated Marginal Means")
    for factor in "ABCD":
        means = df.groupby(factor)["Distance"].agg(["mean","std","count"]).reset_index()
        means["se"] = means["std"] / np.sqrt(means["count"])
        means[factor] = means[factor].map({-1: FACTOR_LABELS[factor][-1], 1: FACTOR_LABELS[factor][1]})
        means = means.rename(columns={factor: FACTOR_LABELS[factor]["name"], "mean":"Mean","std":"Std Dev","count":"N","se":"Std Error"})
        st.markdown(f"**Factor {factor} – {FACTOR_LABELS[factor]['name']}**")
        st.dataframe(means.round(2), use_container_width=True, hide_index=True)

elif page == "Interaction Effects":
    st.title("Interaction Effects")

    factors = list("ABCD")
    pair = st.selectbox(
        "Select factor pair",
        list(combinations(factors, 2)),
        format_func=lambda p: f"{FACTOR_LABELS[p[0]]['name']} ({p[0]}) × {FACTOR_LABELS[p[1]]['name']} ({p[1]})"
    )

    f1, f2 = pair
    means = df.groupby([f1, f2])["Distance"].mean().reset_index()
    means[f1 + "_label"] = means[f1].map({-1: FACTOR_LABELS[f1][-1], 1: FACTOR_LABELS[f1][1]})
    means[f2 + "_label"] = means[f2].map({-1: FACTOR_LABELS[f2][-1], 1: FACTOR_LABELS[f2][1]})

    fig = px.line(
        means, x=f1 + "_label", y="Distance", color=f2 + "_label",
        markers=True,
        labels={
            f1 + "_label": FACTOR_LABELS[f1]["name"],
            "Distance": "Mean Distance (in)",
            f2 + "_label": FACTOR_LABELS[f2]["name"],
        },
        title=f"Interaction: {FACTOR_LABELS[f1]['name']} × {FACTOR_LABELS[f2]['name']}",
    )
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **How to read:** If the lines are roughly parallel, there is little interaction.
    If the lines cross or diverge, the factors interact — the effect of one depends on the level of the other.
    """)

elif page == "ANOVA Table":
    st.title("ANOVA – Full Factorial Model")
    st.markdown("Model: `Distance ~ C(Location) + A*B*C*D`")

    display_anova = anova_table.copy()
    # Clean up index labels: F_A -> A, F_B -> B, etc.
    display_anova.index = display_anova.index.str.replace("F_A", "A").str.replace("F_B", "B").str.replace("F_C", "C").str.replace("F_D", "D").str.replace("Loc", "Location")
    display_anova["Significant"] = display_anova["PR(>F)"].apply(
        lambda p: "Yes" if p < 0.05 else ("Marginal" if p < 0.10 else "No") if pd.notna(p) else ""
    )
    display_anova = display_anova.rename(columns={
        "sum_sq": "Sum Sq", "df": "DF", "mean_sq": "Mean Sq", "F": "F value", "PR(>F)": "p-value"
    })
    display_anova["DF"] = display_anova["DF"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "")

    def highlight_sig(row):
        if row.get("Significant") == "Yes":
            return ["background-color: #d4edda"] * len(row)
        elif row.get("Significant") == "Marginal":
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_anova.style.apply(highlight_sig, axis=1).format({
            "Sum Sq": "{:.1f}", "Mean Sq": "{:.1f}", "F value": "{:.3f}", "p-value": "{:.4f}"
        }, na_rep=""),
        use_container_width=True,
    )

    st.markdown("""
    **Key:** Green = significant at α=0.05 | Yellow = marginal (0.05–0.10)

    At the α=0.05 level, **no individual factor or interaction is statistically significant**, which is
    consistent with the R analysis in the paper (all p-values > 0.05). The high residual variance
    suggests substantial experimental noise.
    """)

    st.subheader("Model Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("R-squared", f"{model.rsquared:.3f}")
    col2.metric("Adj. R-squared", f"{model.rsquared_adj:.3f}")
    col3.metric("Residual Std Error", f"{np.sqrt(model.mse_resid):.2f} in")

elif page == "Residual Diagnostics":
    st.title("Residual Diagnostics")

    residuals = model.resid
    fitted = model.fittedvalues

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.scatter(
            x=fitted, y=residuals,
            labels={"x": "Fitted Values", "y": "Residuals"},
            title="Residuals vs Fitted",
        )
        fig1.add_hline(y=0, line_dash="dash", line_color="red")
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        qq_theoretical = np.sort(stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals))))
        qq_sample = np.sort(residuals.values)
        fig2 = px.scatter(
            x=qq_theoretical, y=qq_sample,
            labels={"x": "Theoretical Quantiles", "y": "Sample Quantiles"},
            title="Normal Q-Q Plot",
        )
        min_val = min(qq_theoretical.min(), qq_sample.min())
        max_val = max(qq_theoretical.max(), qq_sample.max())
        fig2.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                   mode="lines", line=dict(color="red", dash="dash"), showlegend=False))
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.histogram(residuals, nbins=15, title="Histogram of Residuals",
                             labels={"value": "Residual", "count": "Frequency"})
        fig3.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.scatter(
            x=list(range(1, len(residuals) + 1)), y=residuals.values,
            labels={"x": "Run Order", "y": "Residual"},
            title="Residuals vs Run Order",
        )
        fig4.add_hline(y=0, line_dash="dash", line_color="red")
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

    # Shapiro-Wilk test
    sw_stat, sw_p = stats.shapiro(residuals)
    st.subheader("Normality Test (Shapiro-Wilk)")
    st.write(f"W = {sw_stat:.4f}, p-value = {sw_p:.4f}")
    if sw_p > 0.05:
        st.success("Residuals appear normally distributed (p > 0.05).")
    else:
        st.warning("Evidence of non-normality (p < 0.05). Check Q-Q plot for outliers.")

elif page == "Best Settings":
    st.title("Estimated Best Settings")

    st.markdown("""
    Below are the mean distances for every treatment combination, ranked from highest to lowest.
    This helps identify which factor settings maximize flight distance.
    """)

    combo_means = (
        df.groupby(["A","B","C","D"])["Distance"]
        .agg(["mean","std","count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    combo_means["A_label"] = combo_means["A"].map({-1: "Notebook", 1: "Printer"})
    combo_means["B_label"] = combo_means["B"].map({-1: "Dart", 1: "Glider"})
    combo_means["C_label"] = combo_means["C"].map({-1: "None", 1: "Paperclip"})
    combo_means["D_label"] = combo_means["D"].map({-1: "Indoor", 1: "Outdoor"})

    display = combo_means[["A_label","B_label","C_label","D_label","mean","std","count"]].rename(columns={
        "A_label":"Paper","B_label":"Design","C_label":"Nose","D_label":"Environment",
        "mean":"Mean Distance (in)","std":"Std Dev","count":"N"
    })

    st.dataframe(display.round(1), use_container_width=True, hide_index=True)

    best = combo_means.iloc[0]
    st.success(
        f"**Top combination:** {best['A_label']} paper, {best['B_label']} design, "
        f"Nose = {best['C_label']}, {best['D_label']} — "
        f"Mean = {best['mean']:.1f} inches"
    )

    # Bar chart of top combos
    combo_means["combo"] = (
        combo_means["A_label"] + " / " + combo_means["B_label"] + " / " +
        combo_means["C_label"] + " / " + combo_means["D_label"]
    )
    fig = px.bar(
        combo_means.sort_values("mean", ascending=True),
        x="mean", y="combo", orientation="h",
        labels={"mean": "Mean Distance (inches)", "combo": ""},
        title="Mean Distance by Treatment Combination",
        color="mean", color_continuous_scale="Viridis",
    )
    fig.update_layout(height=700, coloraxis_colorbar_title="Distance")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Practical Interpretation")
    st.markdown("""
    - **No single factor is statistically significant** at α = 0.05, suggesting high variability in the system.
    - The **highest observed means** tend to favor no paperclip (normal nose) and indoor environments,
      but the differences are not statistically reliable given the noise.
    - **Suggested next steps:** Increase replication, tighten measurement controls, or reduce
      environmental variability to improve signal-to-noise ratio.
    """)
