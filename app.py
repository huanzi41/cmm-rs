from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cardiometabolic Multimorbidity Risk Prediction",
    page_icon="❤️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

COEF_FILE = BASE_DIR / "cox_coefficients.csv"
RANGE_FILE = BASE_DIR / "var_cp_all2_0.9.csv"
CUT_FILE = BASE_DIR / "risk_cut.csv"

CONTINUOUS_VARIABLES = {"Age"}

CATEGORICAL_GROUPS = {
    "Education": [
        "Education1",
        "Education2",
        "Education3",
        "Education4",
        "Education5",
    ],
    "ADL": [
        "ADL2",
        "ADL3",
        "ADL4",
    ],
}

EDUCATION_OPTIONS = {
    "Illiterate/semi-literate": "Education1",
    "Primary": "Education2",
    "Middle": "Education3",
    "High": "Education4",
    "Vocational": "Education4",
    "College or above": "Education5",
}

EDUCATION_DISPLAY_NAMES = {
    "Education1": "Illiterate/semi-literate",
    "Education2": "Primary",
    "Education3": "Middle",
    "Education4": "High/vocational",
    "Education5": "College or above",
}

ADL_OPTIONS = {
    "Independent": None,
    "Mild": "ADL2",
    "Moderate": "ADL3",
    "Complete": "ADL4",
}

ADL_DISPLAY_NAMES = {
    "ADL2": "Mild",
    "ADL3": "Moderate",
    "ADL4": "Complete",
}

def read_model_file(path):
    return pd.read_excel(path, engine="openpyxl")

def clean_dataframe(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    return df

@st.cache_data
def load_model():
    coef_df = clean_dataframe(read_model_file(COEF_FILE))
    range_df = clean_dataframe(read_model_file(RANGE_FILE))
    cut_df = clean_dataframe(read_model_file(CUT_FILE))

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

    numeric_cut_values = pd.to_numeric(
        cut_df.stack(),
        errors="coerce",
    ).dropna()
    risk_cut = float(numeric_cut_values.iloc[0])

    return coef_df, range_df, risk_cut

def get_range_row(variable, range_df):
    matches = range_df.loc[range_df["Variable"].eq(variable)]
    return None if matches.empty else matches.iloc[0]

def get_display_name(variable, range_df):
    row = get_range_row(variable, range_df)

    if row is not None and pd.notna(row["var_name2"]):
        name = str(row["var_name2"]).strip()
        if name:
            return name

    return {
        "Age": "Age",
        "Sex": "Sex",
        "marital_status": "Marital status",
        "SRH": "Self-rated health",
        "CMM_counts2": "Cardiometabolic condition count",
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
        "hdl": "HDL cholesterol",
        "bun": "Blood urea nitrogen",
        "smoking_status3": "Smoking status",
        "ADL2": "ADL: Mild",
        "ADL3": "ADL: Moderate",
        "ADL4": "ADL: Complete",
        "Education1": "Education: Illiterate/semi-literate",
        "Education2": "Education: Primary",
        "Education3": "Education: Middle",
        "Education4": "Education: High/vocational",
        "Education5": "Education: College or above",
    }.get(variable, variable)

def get_default_value(variable, range_df):
    row = get_range_row(variable, range_df)

    if row is None or pd.isna(row["start_value"]):
        return 0.0

    return float(row["start_value"])

def get_user_inputs(range_df):
    raw_values = {}
    display_values = {}

    st.subheader("Demographic characteristics")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        raw_values["Age"] = float(
            st.number_input(
                "Age (years)",
                min_value=0,
                max_value=120,
                value=70,
                step=1,
            )
        )

    with c2:
        sex = st.selectbox("Sex", ["Female", "Male"])
        raw_values["Sex"] = int(sex == "Male")

    with c3:
        education = st.selectbox(
            "Education",
            list(EDUCATION_OPTIONS.keys()),
        )

        for variable in CATEGORICAL_GROUPS["Education"]:
            raw_values[variable] = 0

        education_variable = EDUCATION_OPTIONS[education]
        raw_values[education_variable] = 1
        display_values["education"] = education

    with c4:
        marital = st.selectbox(
            "Marital status",
            ["Partnered", "Unpartnered"],
        )
        raw_values["marital_status"] = int(marital == "Unpartnered")

    st.subheader("Health behaviors and functional status")
    c1, c2, c3 = st.columns(3)

    with c1:
        smoking = st.selectbox(
            "Smoking history",
            ["Never-smoker", "Ex-smoker", "Current smoker"],
        )
        raw_values["smoking_status3"] = int(smoking == "Current smoker")
        display_values["smoking"] = smoking

    with c2:
        adl = st.selectbox(
            "ADL",
            list(ADL_OPTIONS.keys()),
        )

        for variable in CATEGORICAL_GROUPS["ADL"]:
            raw_values[variable] = 0

        adl_variable = ADL_OPTIONS[adl]
        if adl_variable is not None:
            raw_values[adl_variable] = 1
        display_values["adl"] = adl

    with c3:
        srh = st.selectbox(
            "Self-rated health",
            ["Optimal", "Suboptimal"],
        )
        raw_values["SRH"] = int(srh == "Suboptimal")

    st.subheader("Cardiometabolic conditions")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    with c2:
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    with c3:
        stroke = st.selectbox("Stroke", ["No", "Yes"])
    with c4:
        heart = st.selectbox("Heart disease", ["No", "Yes"])

    condition_count = sum(
        [
            hypertension == "Yes",
            diabetes == "Yes",
            stroke == "Yes",
            heart == "Yes",
        ]
    )

    st.metric("Multimorbidity count", condition_count)

    if condition_count < 2:
        st.error(
            "The multimorbidity count must be at least 2. "
            "Please select at least two cardiometabolic conditions."
        )

    # count = 2 编码为0；count >= 3 编码为1
    raw_values["CMM_counts2"] = int(condition_count >= 3)

    st.subheader("Physical examination")
    c1, c2, c3 = st.columns(3)

    with c1:
        height = st.number_input(
            "Height",
            min_value=50.0,
            max_value=250.0,
            value=165.0,
            step=0.01,
            format="%.2f",
        )

    with c2:
        weight = st.number_input(
            "Weight",
            min_value=20.0,
            max_value=250.0,
            value=65.0,
            step=0.01,
            format="%.2f",
        )

    raw_values["BMI"] = float(weight / ((height / 100.0) ** 2))

    with c3:
        st.metric(
            "Calculated BMI",
            f"{raw_values['BMI']:.2f}",
        )

    st.subheader("Clinical and laboratory measurements")
    ordered_vars = [
        "SP",
        "DP",
        "hb",
        "wbc",
        "plt",
        "fbg",
        "scr",
        "tc",
        "tg",
        "ldl",
        "hdl",
        "bun",
    ]

    cols = st.columns(3)

    for index, variable in enumerate(ordered_vars):
        with cols[index % 3]:
            label = get_display_name(variable, range_df)
            default = get_default_value(variable, range_df)

            if variable in {"SP", "DP"}:
                raw_values[variable] = float(
                    st.number_input(
                        label,
                        min_value=0,
                        max_value=300,
                        value=int(round(default)),
                        step=1,
                    )
                )
            else:
                raw_values[variable] = float(
                    st.number_input(
                        label,
                        min_value=0.0,
                        value=default,
                        step=0.01,
                        format="%.2f",
                    )
                )

    raw_values["_education_label"] = display_values["education"]
    raw_values["_smoking_label"] = display_values["smoking"]
    raw_values["_adl_label"] = display_values["adl"]

    return raw_values, condition_count

def calculate_score(raw_values, coef_df, range_df):
    """
    每个变量的贡献为：

        contribution = coefficient * model_value - center_value

    Age 使用原始连续值；Education 和 ADL 分别累加所有哑变量贡献。
    """
    score = 0.0
    details = []
    processed_variables = set()
    coefficient_map = coef_df.set_index("Variable")

    def get_variable_contribution(variable):
        if variable not in coefficient_map.index:
            return 0.0, ""

        coefficient = float(
            coefficient_map.loc[variable, "Coefficient"]
        )
        center_value = float(
            coefficient_map.loc[variable, "center_value"]
        )
        raw_value = float(raw_values.get(variable, 0.0))
        range_row = get_range_row(variable, range_df)

        if variable in CONTINUOUS_VARIABLES:
            model_value = raw_value
            condition = f"{raw_value:.0f} years"

        elif range_row is not None:
            low = range_row["Low"]
            high = range_row["High"]

            if pd.notna(low) and raw_value < float(low):
                model_value = 1.0
                condition = f"{raw_value:.2f} (< {float(low):.2f})"
            elif pd.notna(high) and raw_value > float(high):
                model_value = 1.0
                condition = f"{raw_value:.2f} (> {float(high):.2f})"
            else:
                model_value = 0.0
                condition = ""

        else:
            model_value = 1.0 if raw_value == 1.0 else 0.0
            condition = "present" if model_value == 1.0 else "absent"

        contribution = coefficient * model_value - center_value
        return contribution, condition

    for group_name, variables in CATEGORICAL_GROUPS.items():
        group_contribution = 0.0

        for variable in variables:
            if variable not in coefficient_map.index:
                continue

            contribution, _ = get_variable_contribution(variable)
            group_contribution += contribution
            processed_variables.add(variable)

        score += group_contribution

        if group_name == "Education":
            selected_label = raw_values.get(
                "_education_label",
                "Unknown",
            )
            display_name = f"Education: {selected_label}"
        else:
            selected_label = raw_values.get(
                "_adl_label",
                "Independent",
            )
            display_name = f"ADL: {selected_label}"

        if group_contribution > 0:
            details.append(
                {
                    "Contributing risk indicator": display_name,
                    "Contribution": group_contribution,
                }
            )

    for entry in coef_df.itertuples(index=False):
        variable = str(entry.Variable).strip()

        if variable in processed_variables:
            continue

        contribution, condition = get_variable_contribution(variable)
        score += contribution

        if contribution > 0:
            if variable == "Age":
                indicator_name = f"Age: {raw_values['Age']:.0f} years"
            elif variable == "smoking_status3":
                smoking_label = raw_values.get(
                    "_smoking_label",
                    "Current smoker",
                )
                indicator_name = f"Smoking status: {smoking_label}"
            elif variable in {"Sex", "marital_status", "SRH"}:
                indicator_name = get_display_name(variable, range_df)
            elif condition:
                display_name = get_display_name(variable, range_df)
                indicator_name = f"{display_name}: {condition}"
            else:
                display_name = get_display_name(variable, range_df)
                indicator_name = f"{display_name}: present"

            details.append(
                {
                    "Contributing risk indicator": indicator_name,
                    "Contribution": contribution,
                }
            )

    details_df = pd.DataFrame(details)

    if not details_df.empty:
        details_df = details_df.sort_values(
            by="Contribution",
            ascending=False,
        ).reset_index(drop=True)

        details_df["Contribution"] = details_df["Contribution"].map(
            lambda value: f"+{value:.4f}"
        )

    return score, details_df

def render_result_cards(score, threshold):
    high_risk = score >= threshold
    risk_label = "High risk" if high_risk else "Low risk"

    if high_risk:
        color = "#B91C1C"
        background = "#FEF2F2"
        border = "#EF4444"
    else:
        color = "#1D4ED8"
        background = "#EFF6FF"
        border = "#3B82F6"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="background:#F8FAFC; border:1px solid #CBD5E1;
                        border-radius:8px; padding:16px;
                        text-align:center; min-height:112px;">
                <div style="font-size:16px; color:#475569;">
                    Risk score
                </div>
                <div style="font-size:30px; font-weight:700;
                            color:#0F172A; margin-top:6px;">
                    {score:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="background:{background}; border:1px solid {border};
                        border-radius:8px; padding:16px;
                        text-align:center; min-height:112px;">
                <div style="font-size:16px; color:{color};">
                    Risk category
                </div>
                <div style="font-size:30px; font-weight:700;
                            color:{color}; margin-top:6px;">
                    {risk_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def main():
    st.title("Cardiometabolic Multimorbidity Risk Prediction")

    coef_df, range_df, risk_cut = load_model()
    user_inputs, condition_count = get_user_inputs(range_df)

    st.divider()

    if st.button(
        "Calculate risk",
        type="primary",
        use_container_width=True,
        disabled=condition_count < 2,
    ):
        final_score, detail_df = calculate_score(
            user_inputs,
            coef_df,
            range_df,
        )

        st.subheader("Prediction result")
        render_result_cards(final_score, risk_cut)

        st.divider()
        st.subheader("Contributing risk indicators")

        if detail_df.empty:
            st.info("No risk-increasing indicators were identified.")
        else:
            st.dataframe(
                detail_df,
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.caption(
        "The calculated risk score and risk category are for reference only "
        "and should not be used as the basis for clinical decision-making."
    )

if __name__ == "__main__":
    main()