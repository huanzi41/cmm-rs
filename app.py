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
    "ldl-c": "LDL cholesterol",
    "hdl": "HDL cholesterol",
    "bun": "Blood urea nitrogen",
    "smoking_status3": "Smoking status",
}

BINARY_INPUT_VARIABLES = {
    "Sex",
    "marital_status",
    "SRH",
    "smoking_status3",
    "CMM_counts2",
}

PHYSICAL_VARIABLES = {
    "BMI",
    "SP",
}

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

@st.cache_data
def load_model():
    coef_df = pd.read_excel(COEF_FILE, engine="openpyxl")
    range_df = pd.read_excel(RANGE_FILE, engine="openpyxl")
    cut_df = pd.read_excel(CUT_FILE, engine="openpyxl")

    coef_df = clean_dataframe(coef_df)
    range_df = clean_dataframe(range_df)
    cut_df = clean_dataframe(cut_df)

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

def clean_dataframe(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    return df

def get_range_row(variable, range_df):
    matches = range_df.loc[range_df["Variable"].eq(variable)]
    return None if matches.empty else matches.iloc[0]

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
    raw_values = {}
    model_variables = get_model_variables(coef_df)

    st.markdown(
        "# ❤️ Cardiometabolic Multimorbidity Risk Prediction"
    )

    st.subheader("Demographic characteristics")
    demographic_variables = [
        variable
        for variable in ["Age", "Sex"]
        if variable in model_variables
    ]
    demographic_columns = st.columns(max(len(demographic_variables), 1))

    if "Age" in model_variables:
        with demographic_columns[0]:
            raw_values["Age"] = float(
                st.number_input(
                    "Age (years)",
                    min_value=0,
                    max_value=120,
                    value=70,
                    step=1,
                )
            )

    if "Sex" in model_variables:
        with demographic_columns[min(1, len(demographic_columns) - 1)]:
            sex = st.selectbox("Sex", ["Female", "Male"])
            raw_values["Sex"] = int(sex == "Male")

    if any(
        variable in model_variables
        for variable in CATEGORICAL_GROUPS["Education"]
    ):
        education = st.selectbox(
            "Education",
            list(EDUCATION_OPTIONS.keys()),
        )

        for variable in CATEGORICAL_GROUPS["Education"]:
            raw_values[variable] = 0

        education_variable = EDUCATION_OPTIONS[education]
        if education_variable in model_variables:
            raw_values[education_variable] = 1

        raw_values["_education_label"] = education

    if "marital_status" in model_variables:
        marital = st.selectbox(
            "Marital status",
            ["Partnered", "Unpartnered"],
        )
        raw_values["marital_status"] = int(marital == "Unpartnered")
        raw_values["_marital_label"] = marital

    st.subheader("Health behaviors and functional status")
    health_variables = []

    if "smoking_status3" in model_variables:
        health_variables.append("smoking_status3")

    if any(
        variable in model_variables
        for variable in CATEGORICAL_GROUPS["ADL"]
    ):
        health_variables.append("ADL")

    if "SRH" in model_variables:
        health_variables.append("SRH")

    health_columns = st.columns(max(len(health_variables), 1))

    health_index = 0

    if "smoking_status3" in model_variables:
        with health_columns[health_index]:
            smoking = st.selectbox(
                "Smoking status",
                ["Never-smoker", "Ex-smoker", "Current smoker"],
            )
            raw_values["smoking_status3"] = int(
                smoking == "Current smoker"
            )
            raw_values["_smoking_label"] = smoking
        health_index += 1

    if "ADL" in health_variables:
        with health_columns[health_index]:
            adl = st.selectbox(
                "ADL",
                list(ADL_OPTIONS.keys()),
            )

            for variable in CATEGORICAL_GROUPS["ADL"]:
                raw_values[variable] = 0

            adl_variable = ADL_OPTIONS[adl]
            if adl_variable in model_variables:
                raw_values[adl_variable] = 1

            raw_values["_adl_label"] = adl
        health_index += 1

    if "SRH" in model_variables:
        with health_columns[health_index]:
            srh = st.selectbox(
                "Self-rated health",
                ["Optimal", "Suboptimal"],
            )
            raw_values["SRH"] = int(srh == "Suboptimal")

    st.subheader("Cardiometabolic conditions")
    condition_columns = st.columns(4)

    with condition_columns[0]:
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    with condition_columns[1]:
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    with condition_columns[2]:
        stroke = st.selectbox("Stroke", ["No", "Yes"])
    with condition_columns[3]:
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

    # Exactly 2 conditions -> 0; 3 or more conditions -> 1
    raw_values["CMM_counts2"] = int(condition_count >= 3)

    st.subheader("Physical examination")
    physical_variables = [
        variable
        for variable in ["BMI", "SP"]
        if variable in model_variables
    ]

    physical_columns = st.columns(max(len(physical_variables) + 2, 1))
    physical_index = 0

    height = None
    weight = None

    with physical_columns[physical_index]:
        height = st.number_input(
            "Height",
            min_value=50.0,
            max_value=250.0,
            value=165.0,
            step=0.01,
            format="%.2f",
        )
    physical_index += 1

    with physical_columns[physical_index]:
        weight = st.number_input(
            "Weight",
            min_value=20.0,
            max_value=250.0,
            value=65.0,
            step=0.01,
            format="%.2f",
        )
    physical_index += 1

    calculated_bmi = float(weight / ((height / 100.0) ** 2))

    if "BMI" in model_variables:
        raw_values["BMI"] = calculated_bmi
        with physical_columns[physical_index]:
            st.metric("BMI", f"{calculated_bmi:.2f}")
        physical_index += 1

    if "SP" in model_variables:
        default_sp = get_default_value("SP", range_df)
        with physical_columns[physical_index]:
            raw_values["SP"] = float(
                st.number_input(
                    get_display_name("SP", range_df),
                    min_value=0,
                    max_value=300,
                    value=int(round(default_sp)),
                    step=1,
                )
            )

    st.subheader("Clinical and laboratory measurements")
    laboratory_variables = [
        variable
        for variable in coef_df["Variable"].tolist()
        if variable in MEASUREMENT_VARIABLES
        and variable in model_variables
        and variable not in PHYSICAL_VARIABLES
    ]

    if laboratory_variables:
        laboratory_columns = st.columns(len(laboratory_variables))

        for index, variable in enumerate(laboratory_variables):
            with laboratory_columns[index]:
                label = get_display_name(variable, range_df)
                default = get_default_value(variable, range_df)

                raw_values[variable] = float(
                    st.number_input(
                        label,
                        min_value=0.0,
                        value=default,
                        step=0.01,
                        format="%.2f",
                    )
                )

    return raw_values, condition_count

def calculate_score(raw_values, coef_df, range_df):
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
                condition = (
                    f"{raw_value:.2f} (< {float(low):.2f})"
                )
            elif pd.notna(high) and raw_value > float(high):
                model_value = 1.0
                condition = (
                    f"{raw_value:.2f} (> {float(high):.2f})"
                )
            else:
                model_value = 0.0
                condition = ""

        else:
            model_value = 1.0 if raw_value == 1.0 else 0.0
            condition = "present" if model_value == 1.0 else "absent"

        contribution = coefficient * model_value - center_value
        return contribution, condition

    # Education and ADL are calculated as grouped dummy-variable terms.
    for group_name, variables in CATEGORICAL_GROUPS.items():
        included_variables = [
            variable
            for variable in variables
            if variable in coefficient_map.index
        ]

        if not included_variables:
            continue

        group_contribution = 0.0

        for variable in included_variables:
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

    # Calculate all remaining variables included in the model.
    for entry in coef_df.itertuples(index=False):
        variable = str(entry.Variable).strip()

        if variable in processed_variables:
            continue

        contribution, condition = get_variable_contribution(variable)
        score += contribution

        if contribution <= 0:
            continue

        if variable == "Age":
            indicator_name = (
                f"Age: {raw_values.get('Age', 0):.0f} years"
            )

        elif variable == "CMM_counts2":
            if raw_values.get("CMM_counts2", 0) == 1:
                indicator_name = (
                    "Cardiometabolic condition count: ≥3 conditions"
                )
            else:
                indicator_name = (
                    "Cardiometabolic condition count: 2 conditions"
                )

        elif variable == "smoking_status3":
            smoking_label = raw_values.get(
                "_smoking_label",
                "Current smoker",
            )
            indicator_name = f"Smoking status: {smoking_label}"

        elif variable == "marital_status":
            marital_label = raw_values.get(
                "_marital_label",
                "Unpartnered",
            )
            indicator_name = f"Marital status: {marital_label}"

        elif variable == "Sex":
            indicator_name = f"Sex: {'Male' if raw_values.get('Sex', 0) == 1 else 'Female'}"

        elif variable == "SRH":
            indicator_name = (
                "Self-rated health: Suboptimal"
                if raw_values.get("SRH", 0) == 1
                else "Self-rated health: Optimal"
            )

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
    coef_df, range_df, risk_cut = load_model()
    user_inputs, condition_count = get_user_inputs(
        coef_df,
        range_df,
    )

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