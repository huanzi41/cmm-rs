from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# 页面设置
# ============================================================
st.set_page_config(
    page_title="Cardiometabolic Multimorbidity Risk Prediction",
    page_icon="❤️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

# cox_coefficients.csv 的实际格式为 Excel 工作簿。
# var_cp_all2_0.9.csv 和 risk_cut.csv 为标准 CSV 文件。
COEF_FILE = BASE_DIR / "cox_coefficients.csv"
RANGE_FILE = BASE_DIR / "var_cp_all2_0.9.csv"
CUT_FILE = BASE_DIR / "risk_cut.csv"

# ============================================================
# 内置文件读取
# ============================================================
@st.cache_data
def load_model():
    """读取内置 Cox 系数、变量范围和风险阈值文件。"""
    try:
        coef_df = pd.read_excel(COEF_FILE, engine="openpyxl")
        range_df = pd.read_csv(RANGE_FILE, encoding="utf-8-sig")
        cut_df = pd.read_csv(CUT_FILE, encoding="utf-8-sig")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Model file not found: {error.filename}. "
            "Please make sure all built-in files are in the same folder as app.py."
        ) from error

    coef_df = coef_df.copy()
    range_df = range_df.copy()
    cut_df = cut_df.copy()

    coef_df.columns = [str(column).strip() for column in coef_df.columns]
    range_df.columns = [str(column).strip() for column in range_df.columns]
    cut_df.columns = [str(column).strip() for column in cut_df.columns]

    # --------------------------------------------------------
    # Cox 系数表
    # 默认前两列分别为变量名与 Cox 系数
    # --------------------------------------------------------
    if coef_df.shape[1] < 2:
        raise ValueError(
            "cox_coefficients.csv must contain at least two columns."
        )

    coef_df = coef_df.iloc[:, :2].copy()
    coef_df.columns = ["Variable", "Coefficient"]

    coef_df["Variable"] = coef_df["Variable"].astype(str).str.strip()
    coef_df["Coefficient"] = pd.to_numeric(
        coef_df["Coefficient"],
        errors="coerce",
    )
    coef_df = coef_df.dropna(subset=["Variable", "Coefficient"])

    center_rows = coef_df.loc[
        coef_df["Variable"].str.lower().eq("center"),
        "Coefficient",
    ]

    if center_rows.empty:
        raise ValueError(
            "cox_coefficients.csv must contain a row named 'center'."
        )

    center = float(center_rows.iloc[0])

    # center 仅用于最终从总分中扣除，不作为普通变量计算。
    coef_df = coef_df.loc[
        ~coef_df["Variable"].str.lower().eq("center")
    ].copy()

    # --------------------------------------------------------
    # 连续变量范围表
    # 需要 Variable、Low、High、var_name2、start value
    # --------------------------------------------------------
    if "Variable" not in range_df.columns:
        range_df = range_df.rename(
            columns={range_df.columns[0]: "Variable"}
        )

    if "Low" not in range_df.columns:
        raise ValueError(
            "var_cp_all2_0.9.csv must contain a 'Low' column."
        )

    if "High" not in range_df.columns:
        raise ValueError(
            "var_cp_all2_0.9.csv must contain a 'High' column."
        )

    if "var_name2" not in range_df.columns:
        range_df["var_name2"] = range_df["Variable"]

    # 兼容 start value 的不同列名写法。
    normalized_columns = {
        str(column).strip().lower().replace("_", " "): column
        for column in range_df.columns
    }

    start_value_column = None
    for candidate in ["start value", "startvalue", "start_value"]:
        if candidate in normalized_columns:
            start_value_column = normalized_columns[candidate]
            break

    if start_value_column is None:
        raise ValueError(
            "var_cp_all2_0.9.csv must contain a 'start value' column."
        )

    range_df = range_df.rename(
        columns={start_value_column: "start_value"}
    )

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

    # --------------------------------------------------------
    # 风险阈值
    # 获取 risk_cut.csv 中第一个有效数值
    # --------------------------------------------------------
    numeric_values = pd.to_numeric(
        cut_df.stack(),
        errors="coerce",
    ).dropna()

    if numeric_values.empty:
        raise ValueError(
            "No valid risk cut-off was found in risk_cut.csv."
        )

    risk_cut = float(numeric_values.iloc[0])

    return coef_df, range_df, center, risk_cut

# ============================================================
# 辅助函数
# ============================================================
def get_range_row(variable, range_df):
    """获取指定变量对应的范围表记录。"""
    matched = range_df.loc[
        range_df["Variable"].astype(str).str.strip().eq(variable)
    ]

    if matched.empty:
        return None

    return matched.iloc[0]

def get_display_name(variable, range_df):
    """优先使用范围表中的 var_name2。"""
    range_row = get_range_row(variable, range_df)

    if range_row is not None:
        display_name = range_row.get("var_name2", variable)

        if pd.notna(display_name) and str(display_name).strip():
            return str(display_name).strip()

    fallback_names = {
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

    return fallback_names.get(variable, variable)

def get_default_value(variable, range_df):
    """
    使用 var_cp_all2_0.9.csv 中的 start value 作为默认值。
    缺失时使用 0.00。
    """
    range_row = get_range_row(variable, range_df)

    if range_row is None:
        return 0.0

    start_value = range_row.get("start_value")

    if pd.notna(start_value):
        return float(start_value)

    return 0.0

def model_contains_variable(variable, coef_df):
    """判断变量是否在 Cox 模型中。"""
    return variable in set(coef_df["Variable"].astype(str))

# ============================================================
# 用户输入
# ============================================================
def get_user_inputs(coef_df, range_df):
    raw_values = {}

    st.subheader("Demographic characteristics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age_default = int(round(get_default_value("Age", range_df)))

        raw_values["Age"] = float(
            st.number_input(
                "Age (years)",
                min_value=0,
                max_value=120,
                value=age_default,
                step=1,
                format="%d",
            )
        )

    with col2:
        sex = st.selectbox(
            "Sex",
            ["Female", "Male"],
        )
        raw_values["Sex"] = int(sex == "Male")

    with col3:
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

        raw_values["Education4"] = int(
            education == "High/vocational"
        )
        raw_values["Education5"] = int(
            education == "College or above"
        )

    with col4:
        marital = st.selectbox(
            "Marital status",
            ["Partnered", "Unpartnered"],
        )
        raw_values["marital_status"] = int(
            marital == "Unpartnered"
        )

    st.subheader("Health behaviors and functional status")

    col1, col2, col3 = st.columns(3)

    with col1:
        smoking = st.selectbox(
            "Smoking history",
            ["Never-smoker", "Ex-smoker", "Current smoker"],
        )
        raw_values["smoking_status3"] = int(
            smoking == "Current smoker"
        )

    with col2:
        adl = st.selectbox(
            "ADL",
            ["Independent", "Mild", "Moderate", "Complete"],
        )
        raw_values["ADL2"] = int(adl == "Mild")
        raw_values["ADL3"] = int(adl == "Moderate")
        raw_values["ADL4"] = int(adl == "Complete")

    with col3:
        self_rated_health = st.selectbox(
            "Self-rated health",
            ["Optimal", "Suboptimal"],
        )
        raw_values["SRH"] = int(
            self_rated_health == "Suboptimal"
        )

    st.subheader("Cardiometabolic conditions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])

    with col2:
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])

    with col3:
        stroke = st.selectbox("Stroke", ["No", "Yes"])

    with col4:
        heart_disease = st.selectbox("Heart disease", ["No", "Yes"])

    raw_values["CMM_counts2"] = float(
        sum(
            [
                hypertension == "Yes",
                diabetes == "Yes",
                stroke == "Yes",
                heart_disease == "Yes",
            ]
        )
    )

    st.subheader("Physical examination")

    col1, col2, col3 = st.columns(3)

    with col1:
        height = st.number_input(
            "Height (cm)",
            min_value=50.00,
            max_value=250.00,
            value=165.00,
            step=0.01,
            format="%.2f",
        )

    with col2:
        weight = st.number_input(
            "Weight (kg)",
            min_value=20.00,
            max_value=250.00,
            value=65.00,
            step=0.01,
            format="%.2f",
        )

    raw_values["BMI"] = float(weight / ((height / 100.0) ** 2))

    with col3:
        st.metric(
            "Calculated BMI (kg/m²)",
            f"{raw_values['BMI']:.2f}",
        )

    st.subheader("Clinical and laboratory measurements")

    possible_continuous_variables = [
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

    continuous_variables = [
        variable
        for variable in possible_continuous_variables
        if (
            get_range_row(variable, range_df) is not None
            or model_contains_variable(variable, coef_df)
        )
    ]

    columns = st.columns(3)

    for index, variable in enumerate(continuous_variables):
        display_name = get_display_name(variable, range_df)
        default_value = get_default_value(variable, range_df)

        with columns[index % 3]:
            if variable in {"SP", "DP"}:
                raw_values[variable] = float(
                    st.number_input(
                        display_name,
                        min_value=0,
                        max_value=300,
                        value=int(round(default_value)),
                        step=1,
                        format="%d",
                    )
                )
            else:
                raw_values[variable] = float(
                    st.number_input(
                        display_name,
                        min_value=0.00,
                        value=float(default_value),
                        step=0.01,
                        format="%.2f",
                    )
                )

    return raw_values

# ============================================================
# 风险评分计算
# ============================================================
def calculate_score(raw_values, coef_df, range_df, center):
    """
    Cox 评分：

    1. 连续变量低于 Low 或高于 High 时，加入对应 Cox 系数；
    2. 分类变量值为 1 时，加入对应 Cox 系数；
    3. 最终风险评分为所有贡献值之和减去 center；
    4. 输出所有实际参与风险评分的变量及其贡献。
    """
    score = -float(center)
    contribution_rows = []

    for row in coef_df.itertuples(index=False):
        variable = str(row.Variable).strip()
        coefficient = float(row.Coefficient)
        value = float(raw_values.get(variable, 0))

        contribution = 0.0
        indicator_text = None
        range_row = get_range_row(variable, range_df)

        # 连续变量：数值位于 Low / High 定义的危险范围时贡献 Cox 系数。
        if range_row is not None:
            low = range_row.get("Low")
            high = range_row.get("High")
            display_name = get_display_name(variable, range_df)

            if pd.notna(low) and value < float(low):
                contribution = coefficient
                indicator_text = (
                    f"{display_name} < {float(low):.2f}"
                )
            elif pd.notna(high) and value > float(high):
                contribution = coefficient
                indicator_text = (
                    f"{display_name} > {float(high):.2f}"
                )

        # 分类变量：哑变量为 1 时贡献 Cox 系数。
        elif value == 1:
            contribution = coefficient
            indicator_text = get_display_name(variable, range_df)

        if contribution != 0:
            score += contribution
            contribution_rows.append(
                {
                    "Contributing indicator": indicator_text,
                    "Contribution": contribution,
                }
            )

    contribution_df = pd.DataFrame(contribution_rows)

    if not contribution_df.empty:
        contribution_df = contribution_df.sort_values(
            by="Contribution",
            key=lambda values: values.abs(),
            ascending=False,
        ).reset_index(drop=True)

        contribution_df["Contribution"] = contribution_df[
            "Contribution"
        ].map(lambda value: f"{value:+.2f}")

    return score, contribution_df

# ============================================================
# 结果卡片
# ============================================================
def show_result_cards(score, risk_cut):
    """展示最终评分和风险分层。"""
    is_high_risk = score >= risk_cut

    if is_high_risk:
        score_background = "#FEF2F2"
        score_border = "#EF4444"
        score_color = "#B91C1C"
        risk_background = "#FEE2E2"
        risk_border = "#DC2626"
        risk_color = "#B91C1C"
        risk_label = "High risk"
    else:
        score_background = "#EFF6FF"
        score_border = "#3B82F6"
        score_color = "#1D4ED8"
        risk_background = "#DBEAFE"
        risk_border = "#2563EB"
        risk_color = "#1D4ED8"
        risk_label = "Low risk"

    score_col, risk_col = st.columns(2)

    with score_col:
        st.markdown(
            f"""
            <div style="
                background-color: {score_background};
                border: 1px solid {score_border};
                border-radius: 8px;
                padding: 24px;
                text-align: center;
            ">
                <div style="
                    color: #4B5563;
                    font-size: 16px;
                    margin-bottom: 8px;
                ">
                    Risk score
                </div>
                <div style="
                    color: {score_color};
                    font-size: 38px;
                    font-weight: 700;
                ">
                    {score:.2f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with risk_col:
        st.markdown(
            f"""
            <div style="
                background-color: {risk_background};
                border: 1px solid {risk_border};
                border-radius: 8px;
                padding: 24px;
                text-align: center;
            ">
                <div style="
                    color: #4B5563;
                    font-size: 16px;
                    margin-bottom: 8px;
                ">
                    Risk cut-off
                </div>
                <div style="
                    color: {risk_color};
                    font-size: 38px;
                    font-weight: 700;
                ">
                    {risk_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# 主程序
# ============================================================
def main():
    st.title("Cardiometabolic Multimorbidity Risk Prediction")
    st.caption("Cox model-based individual risk score calculator")

    try:
        coef_df, range_df, center, risk_cut = load_model()
    except Exception as error:
        st.error(f"Unable to load built-in model files: {error}")
        st.stop()

    raw_values = get_user_inputs(coef_df, range_df)

    st.divider()

    if st.button(
        "Calculate risk",
        type="primary",
        use_container_width=True,
    ):
        score, contribution_df = calculate_score(
            raw_values=raw_values,
            coef_df=coef_df,
            range_df=range_df,
            center=center,
        )

        st.subheader("Prediction result")
        show_result_cards(score, risk_cut)

        st.caption(
            "Reference only: This risk score and classification are intended "
            "for research and educational purposes only. They do not replace "
            "clinical assessment, diagnosis, or medical advice."
        )

        st.divider()
        st.subheader("Contributing risk indicators")

        if contribution_df.empty:
            st.info(
                "No model-defined contributing indicators were identified."
            )
        else:
            st.dataframe(
                contribution_df,
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.caption(
        "Reference only: This calculator must not be used as the sole basis "
        "for clinical decision-making."
    )

if __name__ == "__main__":
    main()