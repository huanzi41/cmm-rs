from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Cardiometabolic Multimorbidity Risk Prediction and Stratification",
    page_icon="❤️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
COEF_FILE = BASE_DIR / "cox_coefficients.csv"
RANGE_FILE = BASE_DIR / "var_cp_all2_0.9.csv"
CUT_FILE = BASE_DIR / "risk_cut.csv"

DISPLAY_RISK_CUTOFF = 1.11
CONTINUOUS_VARIABLES = {"Age"}

CATEGORICAL_GROUPS = {
    "Education": [
        "Education1",
        "Education2",
        "Education3",
        "Education4",
        "Education5",
    ],
    "ADL": ["ADL2", "ADL3", "ADL4"],
}

EDUCATION_OPTIONS = {
    "Illiterate": "Education1",
    "Semi-literate": "Education1",
    "Primary": "Education2",
    "Middle": "Education3",
    "High": "Education4",
    "Vocational": "Education4",
    "College or above": "Education5",
}

ADL_OPTIONS = {
    "Independent": None,
    "Mild": "ADL2",
    "Moderate": "ADL3",
    "Complete": "ADL4",
}

VARIABLE_LABELS = {
    "Age": "Age",
    "Sex": "Sex",
    "marital_status": "Marital status",
    "SRH": "Self-rated health",
    "CMM_counts2": "Multimorbidity count",
    "BMI": "BMI",
    "SP": "Systolic blood pressure",
    "DP": "Diastolic blood pressure",
    "hb": "Haemoglobin",
    "wbc": "White blood cell count",
    "plt": "Platelet count",
    "fbg": "Fasting blood glucose",
    "scr": "Serum creatinine",
    "tc": "Total cholesterol",
    "tg": "Triglycerides",
    "ldl": "LDL cholesterol",
    "ldl-c": "LDL cholesterol",
    "hdl": "HDL cholesterol",
    "bun": "Blood urea nitrogen",
    "smoking_status3": "Smoking status",
}

PHYSICAL_VARIABLES = {"BMI", "SP"}

MEASUREMENT_VARIABLES = {
    "DP",
    "hb",
    "wbc",
    "plt",
    "fbg",
    "scr",
    "tc",
    "tg",
    "ldl",
    "ldl-c",
    "hdl",
    "bun",
}

def clean_dataframe(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    return df

@st.cache_data
def load_model():
    coef_df = clean_dataframe(pd.read_excel(COEF_FILE, engine="openpyxl"))
    range_df = clean_dataframe(pd.read_excel(RANGE_FILE, engine="openpyxl"))
    cut_df = clean_dataframe(pd.read_excel(CUT_FILE, engine="openpyxl"))

    variable_col = next(
        (
            col
            for col in coef_df.columns
            if col.lower() in {"variable", "var_name"}
        ),
        coef_df.columns[0],
    )

    coefficient_col = next(
        (
            col
            for col in coef_df.columns
            if col.lower() in {"coefficient", "coef", "beta"}
        ),
        coef_df.columns[1],
    )

    center_col = next(
        (
            col
            for col in coef_df.columns
            if col.lower() in {"center_value", "center", "mean"}
        ),
        None,
    )

    coef_df = coef_df.rename(
        columns={
            variable_col: "Variable",
            coefficient_col: "Coefficient",
        }
    )

    if center_col is None:
        coef_df["center_value"] = 0.0
    else:
        coef_df = coef_df.rename(columns={center_col: "center_value"})

    coef_df["Variable"] = coef_df["Variable"].astype(str).str.strip()
    coef_df["Coefficient"] = pd.to_numeric(
        coef_df["Coefficient"],
        errors="coerce",
    )
    coef_df["center_value"] = pd.to_numeric(
        coef_df["center_value"],
        errors="coerce",
    ).fillna(0.0)

    coef_df = coef_df.dropna(subset=["Variable", "Coefficient"])
    coef_df = coef_df.loc[
        ~coef_df["Variable"].str.lower().isin(
            {"center", "center_value"}
        )
    ].copy()

    range_df = range_df.iloc[:, :5].copy()
    range_df.columns = [
        "Variable",
        "Low",
        "High",
        "var_name2",
        "start_value",
    ]

    range_df["Variable"] = range_df["Variable"].astype(str).str.strip()
    range_df["var_name2"] = range_df["var_name2"].fillna(
        range_df["Variable"]
    )
    range_df["var_name2"] = range_df["var_name2"].astype(str).str.strip()
    range_df["Low"] = pd.to_numeric(range_df["Low"], errors="coerce")
    range_df["High"] = pd.to_numeric(range_df["High"], errors="coerce")
    range_df["start_value"] = pd.to_numeric(
        range_df["start_value"],
        errors="coerce",
    )

    cut_values = pd.to_numeric(
        cut_df.stack(),
        errors="coerce",
    ).dropna()

    if cut_values.empty:
        raise ValueError("No valid risk cutoff was found in risk_cut.csv.")

    return coef_df, range_df, float(cut_values.iloc[0])

def get_range_row(variable, range_df):
    rows = range_df.loc[range_df["Variable"].eq(variable)]
    return None if rows.empty else rows.iloc[0]

def get_model_variables(coef_df):
    return set(coef_df["Variable"].astype(str).str.strip())

def get_display_name(variable, range_df):
    row = get_range_row(variable, range_df)

    if row is not None and pd.notna(row["var_name2"]):
        name = str(row["var_name2"]).strip()
        if name and name.lower() != "nan":
            return name

    return VARIABLE_LABELS.get(variable, variable)

def get_default_value(variable, range_df):
    row = get_range_row(variable, range_df)

    if row is None or pd.isna(row["start_value"]):
        return 0.0

    return float(row["start_value"])

def get_user_inputs(coef_df, range_df):
    raw = {}
    model_vars = get_model_variables(coef_df)

    st.markdown("# ❤️ Cardiometabolic Multimorbidity Risk Prediction")

    st.subheader("Demographic characteristics")
    demo = st.columns(4)

    with demo[0]:
        if "Age" in model_vars:
            raw["Age"] = float(
                st.number_input(
                    "Age (years)",
                    min_value=0,
                    max_value=120,
                    value=70,
                    step=1,
                )
            )
        else:
            raw["Age"] = 70.0

    with demo[1]:
        sex = st.selectbox("Sex", ["Female", "Male"])
        raw["Sex"] = int(sex == "Male") if "Sex" in model_vars else 0

    with demo[2]:
        education = st.selectbox(
            "Education",
            list(EDUCATION_OPTIONS),
        )

        for variable in CATEGORICAL_GROUPS["Education"]:
            raw[variable] = 0

        selected = EDUCATION_OPTIONS[education]
        if selected in model_vars:
            raw[selected] = 1

        raw["_education_label"] = education

    with demo[3]:
        marital = st.selectbox(
            "Marital status",
            ["Partnered", "Unpartnered"],
        )
        raw["marital_status"] = (
            int(marital == "Unpartnered")
            if "marital_status" in model_vars
            else 0
        )
        raw["_marital_label"] = marital

    st.subheader("Health behaviors and functional status")
    health = st.columns(3)

    with health[0]:
        smoking = st.selectbox(
            "Smoking status",
            ["Never-smoker", "Ex-smoker", "Current smoker"],
        )
        raw["smoking_status3"] = (
            int(smoking == "Current smoker")
            if "smoking_status3" in model_vars
            else 0
        )
        raw["_smoking_label"] = smoking

    with health[1]:
        adl = st.selectbox("ADL", list(ADL_OPTIONS))

        for variable in CATEGORICAL_GROUPS["ADL"]:
            raw[variable] = 0

        selected = ADL_OPTIONS[adl]
        if selected in model_vars:
            raw[selected] = 1

        raw["_adl_label"] = adl

    with health[2]:
        srh = st.selectbox(
            "Self-rated health",
            ["Optimal", "Suboptimal"],
        )
        raw["SRH"] = int(srh == "Suboptimal") if "SRH" in model_vars else 0
        raw["_srh_label"] = srh

    st.subheader("Cardiometabolic conditions")
    conditions = st.columns(5)
    labels = ["Hypertension", "Diabetes", "Stroke", "Heart disease"]
    selected_conditions = []

    for index, label in enumerate(labels):
        with conditions[index]:
            value = st.selectbox(
                label,
                ["No", "Yes"],
                key=f"condition_{index}",
            )
            selected_conditions.append(value == "Yes")

    condition_count = int(sum(selected_conditions))

    with conditions[4]:
        st.metric("Multimorbidity count", condition_count)

    if condition_count < 2:
        st.error(
            "The multimorbidity count must be at least 2. "
            "Please select at least two cardiometabolic conditions."
        )

    raw["CMM_counts2"] = int(condition_count >= 3)
    raw["_condition_count"] = condition_count

    st.subheader("Physical examination")
    physical = st.columns(4)

    with physical[0]:
        height = st.number_input(
            "Height",
            min_value=50.0,
            max_value=250.0,
            value=165.0,
            step=0.01,
            format="%.2f",
        )

    with physical[1]:
        weight = st.number_input(
            "Weight",
            min_value=20.0,
            max_value=250.0,
            value=65.0,
            step=0.01,
            format="%.2f",
        )

    bmi = float(weight / ((height / 100.0) ** 2))

    if "BMI" in model_vars:
        raw["BMI"] = bmi

    with physical[2]:
        st.metric("BMI", f"{bmi:.2f}")

    if "SP" in model_vars:
        with physical[3]:
            raw["SP"] = float(
                st.number_input(
                    get_display_name("SP", range_df),
                    min_value=0,
                    max_value=300,
                    value=int(round(get_default_value("SP", range_df))),
                    step=1,
                )
            )

    laboratory_vars = [
        variable
        for variable in coef_df["Variable"]
        if variable in MEASUREMENT_VARIABLES
        and variable not in PHYSICAL_VARIABLES
    ]

    if laboratory_vars:
        st.subheader("Clinical and laboratory measurements")
        lab_columns = st.columns(len(laboratory_vars))

        for index, variable in enumerate(laboratory_vars):
            with lab_columns[index]:
                raw[variable] = float(
                    st.number_input(
                        get_display_name(variable, range_df),
                        min_value=0.0,
                        value=get_default_value(variable, range_df),
                        step=0.01,
                        format="%.2f",
                    )
                )

    return raw, condition_count

def calculate_score(raw, coef_df, range_df):
    score = 0.0
    details = []
    processed = set()
    coefficient_map = coef_df.set_index("Variable")

    def contribution_for(variable):
        if variable not in coefficient_map.index:
            return 0.0, ""

        coefficient = float(coefficient_map.loc[variable, "Coefficient"])
        center = float(coefficient_map.loc[variable, "center_value"])
        value = float(raw.get(variable, 0.0))
        range_row = get_range_row(variable, range_df)

        if variable in CONTINUOUS_VARIABLES:
            model_value = value
            condition = f"{value:.0f} years"
        elif range_row is not None:
            low = range_row["Low"]
            high = range_row["High"]

            if pd.notna(low) and value < float(low):
                model_value = 1.0
                condition = f"{value:.2f} (< {float(low):.2f})"
            elif pd.notna(high) and value > float(high):
                model_value = 1.0
                condition = f"{value:.2f} (> {float(high):.2f})"
            else:
                model_value = 0.0
                condition = ""
        else:
            model_value = float(value == 1.0)
            condition = "present" if model_value else "absent"

        return coefficient * model_value - center, condition

    for group_name, variables in CATEGORICAL_GROUPS.items():
        included = [
            variable
            for variable in variables
            if variable in coefficient_map.index
        ]

        if not included:
            continue

        group_score = 0.0

        for variable in included:
            contribution, _ = contribution_for(variable)
            group_score += contribution
            processed.add(variable)

        score += group_score

        if group_score > 0:
            label_key = f"_{group_name.lower()}_label"
            label = raw.get(label_key, "Unknown")

            details.append(
                {
                    "Risk indicator": f"{group_name}: {label}",
                    "Contribution": group_score,
                }
            )

    for variable in coef_df["Variable"]:
        if variable in processed:
            continue

        contribution, condition = contribution_for(variable)
        score += contribution

        if contribution <= 0:
            continue

        if variable == "Age":
            name = f"Age: {raw.get('Age', 0):.0f} years"
        elif variable == "CMM_counts2":
            name = "Multimorbidity count >= 3"
        elif variable == "smoking_status3":
            name = (
                "Smoking status: "
                f"{raw.get('_smoking_label', 'Current smoker')}"
            )
        elif variable == "marital_status":
            name = (
                "Marital status: "
                f"{raw.get('_marital_label', 'Unpartnered')}"
            )
        elif variable == "Sex":
            name = f"Sex: {'Male' if raw.get('Sex', 0) else 'Female'}"
        elif variable == "SRH":
            name = (
                "Self-rated health: "
                f"{raw.get('_srh_label', 'Suboptimal')}"
            )
        elif condition:
            name = f"{get_display_name(variable, range_df)}: {condition}"
        else:
            name = f"{get_display_name(variable, range_df)}: present"

        details.append(
            {
                "Risk indicator": name,
                "Contribution": contribution,
            }
        )

    details_df = pd.DataFrame(details)

    if not details_df.empty:
        details_df = details_df.sort_values(
            "Contribution",
            ascending=False,
        ).reset_index(drop=True)

    return score, details_df

def render_result_cards(score, threshold):
    low_risk = score <= threshold
    risk_label = "Low risk" if low_risk else "High risk"

    if low_risk:
        color = "#1D4ED8"
        background = "#EFF6FF"
        border = "#3B82F6"
    else:
        color = "#B91C1C"
        background = "#FEF2F2"
        border = "#EF4444"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="background:#F8FAFC;border:1px solid #CBD5E1;
                        border-radius:8px;padding:16px;text-align:center;
                        min-height:112px;">
                <div style="font-size:16px;color:#475569;">
                    Risk score
                </div>
                <div style="font-size:30px;font-weight:700;
                            color:#0F172A;margin-top:6px;">
                    {score:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="background:{background};border:1px solid {border};
                        border-radius:8px;padding:16px;text-align:center;
                        min-height:112px;">
                <div style="font-size:16px;color:{color};">
                    Risk category
                </div>
                <div style="font-size:30px;font-weight:700;
                            color:{color};margin-top:6px;">
                    {risk_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Risk assessment is classified into two categories: "
        f"low risk (<={DISPLAY_RISK_CUTOFF:.2f}) and "
        f"high risk (>{DISPLAY_RISK_CUTOFF:.2f}). "
        "The value 1.11 represents the 90th percentile of the risk score "
        "among older adults with cardiometabolic multimorbidity in Shenzhen."
    )

def render_contribution_chart(detail_df):
    plot_df = detail_df.sort_values(
        "Contribution",
        ascending=True,
    ).copy()

    labels = [
        f"+{value:.4f}" if value >= 0 else f"{value:.4f}"
        for value in plot_df["Contribution"]
    ]

    colors = [
        "#DC2626" if value > 0 else "#2563EB"
        for value in plot_df["Contribution"]
    ]

    figure_height = max(260, min(720, 100 + len(plot_df) * 48))

    max_contribution = float(plot_df["Contribution"].max())
    right_margin = max(0.05, max_contribution * 0.20)

    fig = go.Figure(
        go.Bar(
            x=plot_df["Contribution"],
            y=plot_df["Risk indicator"],
            orientation="h",
            width=0.42,
            marker_color=colors,
            text=labels,
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Contribution: %{x:.4f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=figure_height,
        margin=dict(l=240, r=100, t=15, b=45),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        showlegend=False,
        bargap=0.45,
        font=dict(
            family="Arial, sans-serif",
            size=13,
            color="#0F172A",
        ),
        xaxis=dict(
            title="Contribution to risk score",
            range=[0, max_contribution + right_margin],
            showgrid=True,
            gridcolor="#E2E8F0",
            zeroline=True,
            zerolinecolor="#94A3B8",
            linecolor="#94A3B8",
            fixedrange=True,
        ),
        yaxis=dict(
            title=None,
            showgrid=False,
            linecolor="#94A3B8",
            categoryorder="array",
            categoryarray=plot_df["Risk indicator"].tolist(),
            tickfont=dict(size=13),
            fixedrange=True,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

def main():
    coef_df, range_df, risk_cut = load_model()
    raw, condition_count = get_user_inputs(coef_df, range_df)

    st.divider()

    if st.button(
        "Calculate risk",
        type="primary",
        use_container_width=True,
        disabled=condition_count < 2,
    ):
        score, detail_df = calculate_score(raw, coef_df, range_df)

        st.subheader("Prediction result")
        render_result_cards(score, risk_cut)

        st.divider()
        st.subheader("Contributing risk indicators")

        if detail_df.empty:
            st.info("No risk-increasing indicators were identified.")
        else:
            render_contribution_chart(detail_df)

    st.divider()
    st.caption(
        "The calculated risk score and risk category are for reference only "
        "and should not be used as the basis for clinical decision-making."
    )

if __name__ == "__main__":
    main()