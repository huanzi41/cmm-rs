from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cardiometabolic Multimorbidity Risk Prediction",
    page_icon="❤️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

# 文件实际内容为 xlsx，即使文件扩展名仍然是 csv
COEF_FILE = BASE_DIR / "cox_coefficients.csv"
RANGE_FILE = BASE_DIR / "var_cp_all2_0.9.csv"
CUT_FILE = BASE_DIR / "risk_cut.csv"

def read_excel_safely(path):
    """读取实际为 xlsx 格式的内置文件，避免按 UTF-8 解析。"""
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
            "cox_coefficients.xlsx must contain Variable and Coefficient columns."
        )

    # 保留 Variable、Coefficient，并寻找 center_value 列
    coef_df = coef_df.copy()
    coef_df.columns = [str(col).strip() for col in coef_df.columns]

    variable_col = next(
        (col for col in coef_df.columns if col.lower() in {"variable", "var_name"}),
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
        # 如果文件没有 center_value，默认中心值为 0
        coef_df["center_value"] = 0.0
    else:
        coef_df = coef_df.rename(columns={center_col: "center_value"})

    coef_df["Variable"] = coef_df["Variable"].astype(str).str.strip()
    coef_df["Coefficient"] = pd.to_numeric(
        coef_df["Coefficient"], errors="coerce"
    )
    coef_df["center_value"] = pd.to_numeric(
        coef_df["center_value"], errors="coerce"
    ).fillna(0.0)

    coef_df = coef_df.dropna(subset=["Variable", "Coefficient"])
    coef_df = coef_df.loc[
        ~coef_df["Variable"].str.lower().isin({"center", "center_value"})
    ].copy()

    if range_df.shape[1] < 5:
        raise ValueError(
            "var_cp_all2_0.9.xlsx must contain Variable, Low, High, "
            "var_name2 and start_value columns."
        )

    range_df = range_df.iloc[:, :5].copy()
    range_df.columns = ["Variable", "Low", "High", "var_name2", "start_value"]
    range_df["Variable"] = range_df["Variable"].astype(str).str.strip()
    range_df["var_name2"] = range_df["var_name2"].fillna(range_df["Variable"])
    range_df["var_name2"] = range_df["var_name2"].astype(str).str.strip()
    range_df["Low"] = pd.to_numeric(range_df["Low"], errors="coerce")
    range_df["High"] = pd.to_numeric(range_df["High"], errors="coerce")
    range_df["start_value"] = pd.to_numeric(
        range_df["start_value"], errors="coerce"
    )

    numeric_cut_values = pd.to_numeric(cut_df.stack(), errors="coerce").dropna()
    if numeric_cut_values.empty:
        raise ValueError("risk_cut.xlsx does not contain a numeric threshold.")

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
        "smoking_status3": "Current smoker",
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
        for level in range(1, 6):
            raw_values[f"Education{level}"] = 0
        education_index = {
            "Illiterate/semi-literate": 1,
            "Primary": 2,
            "Middle": 3,
            "High/vocational": 4,
            "College or above": 5,
        }[education]
        raw_values[f"Education{education_index}"] = 1

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
        for level in range(2, 5):
            raw_values[f"ADL{level}"] = int(adl == {2: "Mild", 3: "Moderate", 4: "Complete"}[level])

    with c3:
        srh = st.selectbox("Self-rated health", ["Optimal", "Suboptimal"])
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
    # 2 个共病为 0，超过 2 个共病为 1；其他情况也按二分类变量处理
    raw_values["CMM_counts2"] = int(condition_count > 2)

    st.subheader("Physical examination")
    c1, c2, c3 = st.columns(3)
    with c1:
        height = st.number_input(
            "Height (cm)", 50.0, 250.0, 165.0, step=0.01, format="%.2f"
        )
    with c2:
        weight = st.number_input(
            "Weight (kg)", 20.0, 250.0, 65.0, step=0.01, format="%.2f"
        )
    raw_values["BMI"] = float(weight / ((height / 100.0) ** 2))
    with c3:
        st.metric("Calculated BMI (kg/m²)", f"{raw_values['BMI']:.2f}")

    st.subheader("Clinical and laboratory measurements")
    ordered_vars = [
        "SP", "DP", "hb", "wbc", "plt", "fbg",
        "scr", "tc", "tg", "ldl", "hdl", "bun",
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

    return raw_values

def calculate_score(raw_values, coef_df, range_df):
    """
    计算规则：
    - 体检和血检变量：超出 [Low, High] 区间时 indicator=1，否则为 0；
    - 二分类变量：直接使用 0/1；
    - 每个变量的线性项为 coefficient * (indicator - center_value)；
    - ADL/Education 作为哑变量，分别使用所选等级对应的 indicator。
    """
    score = 0.0
    details = []

    for entry in coef_df.itertuples(index=False):
        variable = str(entry.Variable).strip()
        coefficient = float(entry.Coefficient)
        center_value = float(entry.center_value)
        raw_value = float(raw_values.get(variable, 0.0))

        range_row = get_range_row(variable, range_df)
        if range_row is not None:
            low = range_row["Low"]
            high = range_row["High"]

            # 有阈值的变量按是否超出区间转换为 risk indicator
            if pd.notna(low) and raw_value < float(low):
                indicator = 1.0
                condition = f"< {float(low):.2f}"
            elif pd.notna(high) and raw_value > float(high):
                indicator = 1.0
                condition = f"> {float(high):.2f}"
            else:
                indicator = 0.0
                condition = "within threshold"
        else:
            indicator = 1.0 if raw_value == 1 else 0.0
            condition = "risk indicator = 1" if indicator else "risk indicator = 0"

        centered_value = indicator - center_value
        contribution = coefficient * centered_value
        score += contribution

        if indicator == 1.0 or centered_value != 0.0:
            details.append(
                {
                    "Contributing indicator": (
                        f"{get_display_name(variable, range_df)} ({condition})"
                    ),
                    "Indicator": indicator,
                    "Center value": center_value,
                    "Contribution": contribution,
                }
            )

    details_df = pd.DataFrame(details)
    if not details_df.empty:
        details_df = details_df.sort_values(
            by="Contribution",
            key=lambda values: values.abs(),
            ascending=False,
        ).reset_index(drop=True)
        details_df["Indicator"] = details_df["Indicator"].map(lambda value: f"{value:.0f}")
        details_df["Center value"] = details_df["Center value"].map(lambda value: f"{value:.4f}")
        details_df["Contribution"] = details_df["Contribution"].map(lambda value: f"{value:+.4f}")

    return score, details_df

def render_result_cards(score, threshold):
    high_risk = score >= threshold
    risk_label = "High risk" if high_risk else "Low risk"
    score_color = "#B91C1C" if high_risk else "#1D4ED8"
    background = "#FEF2F2" if high_risk else "#EFF6FF"
    border = "#EF4444" if high_risk else "#3B82F6"

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk score", f"{score:.4f}")
    with col2:
        st.metric("Risk cut-off", f"{threshold:.4f}")

    st.markdown(
        f"<div style='background:{background}; border:1px solid {border}; "
        f"border-radius:8px; padding:16px; color:{score_color}; "
        f"font-size:24px; font-weight:700; text-align:center'>{risk_label}</div>",
        unsafe_allow_html=True,
    )

def main():
    st.title("Cardiometabolic Multimorbidity Risk Prediction")
    st.caption("Cox model-based individual risk score calculator")

    try:
        coef_df, range_df, risk_cut = load_model()
    except (RuntimeError, ValueError) as error:
        st.error(str(error))
        st.stop()

    user_inputs = get_user_inputs(coef_df, range_df)
    st.divider()

    if st.button("Calculate risk", type="primary", use_container_width=True):
        final_score, detail_df = calculate_score(
            user_inputs,
            coef_df,
            range_df,
        )

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
            st.info("No model-defined contributing indicators were identified.")
        else:
            st.dataframe(detail_df, use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "Reference only: This calculator is intended for research and educational "
        "use only and must not be used as the sole basis for clinical decision-making."
    )

if __name__ == "__main__":
    main()