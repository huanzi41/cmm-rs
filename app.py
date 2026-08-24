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

def read_excel_safely(path):
    """所有内置文件真实格式都是 .xlsx，使用 xlrd/openpyxl 方式读取，绕过 UTF-8 解码错误。"""
    return pd.read_excel(str(path), engine="openpyxl")

def clean_dataframe(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df

@st.cache_data
def load_model():
    try:
        coef_df = clean_dataframe(read_excel_safely(COEF_FILE))
        range_df = clean_dataframe(read_excel_safely(RANGE_FILE))
        cut_df = clean_dataframe(read_excel_safely(CUT_FILE))
    except Exception as error:
        raise RuntimeError(
            f"Unable to load built-in model files: {error}"
        ) from error

    if coef_df.shape[1] < 2:
        raise ValueError(
            "cox_coefficients.csv must contain at least 2 columns: Variable and Coefficient."
        )

    coef_df = coef_df.iloc[:, :2].copy()
    coef_df.columns = ["Variable", "Coefficient"]
    coef_df["Variable"] = coef_df["Variable"].astype(str).str.strip()
    coef_df["Coefficient"] = pd.to_numeric(coef_df["Coefficient"], errors="coerce")
    coef_df = coef_df.dropna(subset=["Variable", "Coefficient"])

    center_rows = coef_df.loc[
        coef_df["Variable"].str.lower() == "center",
        "Coefficient",
    ]
    if center_rows.empty:
        raise ValueError(
            "cox_coefficients.csv must contain a row named 'center'."
        )
    center = float(center_rows.iloc[0])
    coef_df = coef_df.loc[
        coef_df["Variable"].str.lower() != "center"
    ].copy()

    if range_df.shape[1] < 5:
        raise ValueError(
            "var_cp_all2_0.9.csv needs 5 fields: Variable, Low, High, var_name2, start_value."
        )

    range_df = range_df.iloc[:, :5].copy()
    range_df.columns = ["Variable", "Low", "High", "var_name2", "start_value"]
    range_df["Variable"] = range_df["Variable"].astype(str).str.strip()
    range_df["var_name2"] = range_df["var_name2"].fillna(range_df["Variable"])
    range_df["var_name2"] = range_df["var_name2"].astype(str).str.strip()
    range_df["Low"] = pd.to_numeric(range_df["Low"], errors="coerce")
    range_df["High"] = pd.to_numeric(range_df["High"], errors="coerce")
    range_df["start_value"] = pd.to_numeric(range_df["start_value"], errors="coerce")

    numeric_cut_values = pd.to_numeric(cut_df.stack(), errors="coerce").dropna()
    if numeric_cut_values.empty:
        raise ValueError(
            "risk_cut.csv does not contain valid numeric risk threshold."
        )
    risk_cut = float(numeric_cut_values.iloc[0])

    return coef_df, range_df, center, risk_cut

def get_range_row(variable, range_df):
    matches = range_df.loc[
        range_df["Variable"].str.strip().eq(variable)
    ]
    return None if matches.empty else matches.iloc[0]

def get_display_name(variable, range_df):
    row = get_range_row(variable, range_df)
    if row is not None:
        name = str(row["var_name2"]).strip()
        if name:
            return name
    fallback_map = {
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
        "smoking_status3": "Smoking history: Current smoker",
        "ADL2": "ADL: Mild",
        "ADL3": "ADL: Moderate",
        "ADL4": "ADL: Complete",
        "Education4": "Education: High/vocational",
        "Education5": "Education: College or above",
    }
    return fallback_map.get(variable, variable)

def get_default_value(variable, range_df):
    row = get_range_row(variable, range_df)
    if row is None:
        return 0.0
    val = row.get("start_value", 0.0)
    return float(val) if pd.notna(val) else 0.0

def get_user_inputs(coef_df, range_df):
    raw_values = {}

    st.subheader("Demographic characteristics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        raw_values["Age"] = float(
            st.number_input(
                "Age (years)",
                min_value=0,
                max_value=120,
                value=int(round(get_default_value("Age", range_df))),
                step=1,
                format="%d",
            )
        )
    with c2:
        sex = st.selectbox("Sex", ["Female", "Male"])
        raw_values["Sex"] = int(sex == "Male")
    with c3:
        education = st.selectbox(
            "Education",
            [
                "Illiterate/semi-literate",
                "Primary",
                "Middle",
                "High/vocational",
                "College or above",
            ],
        )
        raw_values["Education4"] = int(education == "High/vocational")
        raw_values["Education5"] = int(education == "College or above")
    with c4:
        marital = st.selectbox("Marital status", ["Partnered", "Unpartnered"])
        raw_values["marital_status"] = int(marital == "Unpartnered")

    st.subheader("Health behaviors and functional status")
    c1, c2, c3 = st.columns(3)
    with c1:
        smoking = st.selectbox(
            "Smoking history",
            ["Never-smoker", "Ex-smoker", "Current smoker"],
        )
        raw_values["smoking_status3"] = int(smoking == "Current smoker")
    with c2:
        adl = st.selectbox("ADL", ["Independent", "Mild", "Moderate", "Complete"])
        raw_values["ADL2"] = int(adl == "Mild")
        raw_values["ADL3"] = int(adl == "Moderate")
        raw_values["ADL4"] = int(adl == "Complete")
    with c3:
        srh = st.selectbox("Self-rated health", ["Optimal", "Suboptimal"])
        raw_values["SRH"] = int(srh == "Suboptimal")

    st.subheader("Cardiometabolic conditions")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ht = st.selectbox("Hypertension", ["No", "Yes"])
    with c2:
        dm = st.selectbox("Diabetes", ["No", "Yes"])
    with c3:
        stroke = st.selectbox("Stroke", ["No", "Yes"])
    with c4:
        hd = st.selectbox("Heart disease", ["No", "Yes"])
    raw_values["CMM_counts2"] = float(sum([
        ht == "Yes",
        dm == "Yes",
        stroke == "Yes",
        hd == "Yes",
    ]))

    st.subheader("Physical examination")
    c1, c2, c3 = st.columns(3)
    with c1:
        height = st.number_input(
            "Height (cm)",
            min_value=50.00,
            max_value=250.00,
            value=165.00,
            step=0.01,
            format="%.2f",
        )
    with c2:
        weight = st.number_input(
            "Weight (kg)",
            min_value=20.00,
            max_value=250.00,
            value=65.00,
            step=0.01,
            format="%.2f",
        )
    raw_values["BMI"] = float(weight / ((height / 100.0) ** 2))
    with c3:
        st.metric("Calculated BMI (kg/m²)", f"{raw_values['BMI']:.2f}")

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
    for idx, var_name in enumerate(ordered_vars):
        with cols[idx % 3]:
            dname = get_display_name(var_name, range_df)
            dval = get_default_value(var_name, range_df)
            if var_name in ("SP", "DP"):
                raw_values[var_name] = float(
                    st.number_input(
                        dname,
                        min_value=0,
                        max_value=300,
                        value=int(round(dval)),
                        step=1,
                        format="%d",
                    )
                )
            else:
                raw_values[var_name] = float(
                    st.number_input(
                        dname,
                        min_value=0.00,
                        value=float(dval),
                        step=0.01,
                        format="%.2f",
                    )
                )

    return raw_values

def calculate_score(raw_values, coef_df, range_df, center):
    score = -float(center)
    details = []
    for entry in coef_df.itertuples(index=False):
        var = str(entry.Variable).strip()
        coef = float(entry.Coefficient)
        value = float(raw_values.get(var, 0))
        contrib = 0.0
        label = None
        rr = get_range_row(var, range_df)

        if rr is not None:
            name = get_display_name(var, range_df)
            low, high = rr["Low"], rr["High"]
            if pd.notna(low) and value < float(low):
                contrib = coef
                label = f"{name} < {float(low):.2f}"
            elif pd.notna(high) and value > float(high):
                contrib = coef
                label = f"{name} > {float(high):.2f}"
        elif value == 1:
            contrib = coef
            label = get_display_name(var, range_df)

        if contrib != 0 and label is not None:
            score += contrib
            details.append({
                "Contributing indicator": label,
                "Contribution": contrib,
            })

    details_df = pd.DataFrame(details)
    if not details_df.empty:
        details_df = details_df.sort_values(
            by="Contribution",
            key=lambda s: s.abs(),
            ascending=False,
        ).reset_index(drop=True)
        details_df["Contribution"] = details_df["Contribution"].map(
            lambda v: f"{v:+.2f}"
        )

    return score, details_df

def render_result_cards(score, threshold):
    high_risk = score >= threshold
    if high_risk:
        s_bg, s_bd, s_c = "#FEF2F2", "#EF4444", "#B91C1C"
        r_bg, r_bd, r_c = "#FEE2E2", "#DC2626", "#B91C1C"
        r_label = "High risk"
    else:
        s_bg, s_bd, s_c = "#EFF6FF", "#3B82F6", "#1D4ED8"
        r_bg, r_bd, r_c = "#DBEAFE", "#2563EB", "#1D4ED8"
        r_label = "Low risk"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div style="
                background:{s_bg};
                border:1px solid {s_bd};
                border-radius:8px;
                padding:24px;
                text-align:center;
            ">
                <div style="color:#4B5563; font-size:16px; margin-bottom:8px;">
                    Risk score
                </div>
                <div style="color:{s_c}; font-size:38px; font-weight:700;">
                    {score:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div style="
                background:{r_bg};
                border:1px solid {r_bd};
                border-radius:8px;
                padding:24px;
                text-align:center;
            ">
                <div style="color:#4B5563; font-size:16px; margin-bottom:8px;">
                    Risk cut-off
                </div>
                <div style="color:{r_c}; font-size:38px; font-weight:700;">
                    {r_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def main():
    st.title("Cardiometabolic Multimorbidity Risk Prediction")
    st.caption("Cox model-based individual risk score calculator")

    try:
        coef_df, range_df, center, risk_cut = load_model()
    except RuntimeError as error:
        st.error(str(error))
        st.stop()

    user_inputs = get_user_inputs(coef_df, range_df)

    st.divider()

    if st.button("Calculate risk", type="primary", use_container_width=True):
        final_score, detail_df = calculate_score(user_inputs, coef_df, range_df, center)

        st.subheader("Prediction result")
        render_result_cards(final_score, risk_cut)

        st.caption(
            "Reference only: The calculated risk score and risk classification "
            "are for research and educational purposes only and do not replace "
            "professional clinical assessment, diagnosis, or medical advice."
        )

        st.divider()
        st.subheader("Contributing risk indicators")

        if detail_df.empty:
            st.info(
                "No model-defined contributing indicators were identified "
                "from the entered values."
            )
        else:
            st.dataframe(
                detail_df,
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.caption(
        "Reference only: This calculator is intended for research and educational "
        "use only and must not be used as the sole basis for clinical decision-making."
    )

if __name__ == "__main__":
    main()