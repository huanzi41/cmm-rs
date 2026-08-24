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

# ============================================================
# 内置模型文件读取
# ============================================================
@st.cache_data
def load_model():
    """
    读取应用内置的固定模型文件：
    - cox_coefficients.csv：Cox 回归系数
    - var_cp_all2_0.9.csv：变量危险范围及展示名称
    - risk_cut.csv：风险分层阈值
    """
    coef_df = pd.read_csv(COEF_FILE)
    range_df = pd.read_csv(RANGE_FILE)
    cut_df = pd.read_csv(CUT_FILE)

    coef_df = coef_df.copy()
    range_df = range_df.copy()
    cut_df = cut_df.copy()

    coef_df.columns = [str(col).strip() for col in coef_df.columns]
    range_df.columns = [str(col).strip() for col in range_df.columns]
    cut_df.columns = [str(col).strip() for col in cut_df.columns]

    # --------------------------------------------------------
    # Cox 系数文件：默认取前两列
    # --------------------------------------------------------
    if coef_df.shape[1] < 2:
        raise ValueError(
            "cox_coefficients.csv 至少应包含两列：变量名和 Cox 系数。"
        )

    coef_df = coef_df.iloc[:, :2].copy()
    coef_df.columns = ["Variable", "Coefficient"]

    coef_df["Variable"] = coef_df["Variable"].astype(str).str.strip()
    coef_df["Coefficient"] = pd.to_numeric(
        coef_df["Coefficient"],
        errors="coerce",
    )

    coef_df = coef_df.dropna(subset=["Variable", "Coefficient"])

    # 读取模型中心化常数
    center_rows = coef_df.loc[
        coef_df["Variable"].str.lower().eq("center"),
        "Coefficient",
    ]

    if center_rows.empty:
        raise ValueError(
            "cox_coefficients.csv 中未找到名为 center 的模型常数。"
        )

    center = float(center_rows.iloc[0])

    # center 不作为预测变量参与后续循环
    coef_df = coef_df.loc[
        ~coef_df["Variable"].str.lower().eq("center")
    ].copy()

    # --------------------------------------------------------
    # 连续变量范围文件
    # 必须保留 var_name2，用于界面展示
    # --------------------------------------------------------
    if range_df.shape[1] < 3:
        raise ValueError(
            "var_cp_all2_0.9.csv 至少应包含 Variable、Low 和 High 三列。"
        )

    # 如果原始文件不存在这些列名，则默认前三列依次是 Variable、Low、High
    if "Variable" not in range_df.columns:
        range_df = range_df.rename(
            columns={range_df.columns[0]: "Variable"}
        )

    if "Low" not in range_df.columns:
        range_df = range_df.rename(
            columns={range_df.columns[1]: "Low"}
        )

    if "High" not in range_df.columns:
        range_df = range_df.rename(
            columns={range_df.columns[2]: "High"}
        )

    if "var_name2" not in range_df.columns:
        range_df["var_name2"] = range_df["Variable"]

    range_df["Variable"] = range_df["Variable"].astype(str).str.strip()
    range_df["var_name2"] = range_df["var_name2"].fillna(
        range_df["Variable"]
    )
    range_df["var_name2"] = range_df["var_name2"].astype(str).str.strip()

    range_df["Low"] = pd.to_numeric(range_df["Low"], errors="coerce")
    range_df["High"] = pd.to_numeric(range_df["High"], errors="coerce")

    # --------------------------------------------------------
    # 风险阈值文件：读取第一个有效数值
    # --------------------------------------------------------
    risk_values = pd.to_numeric(
        cut_df.stack(),
        errors="coerce",
    ).dropna()

    if risk_values.empty:
        raise ValueError("risk_cut.csv 中未找到有效的风险阈值。")

    risk_cut = float(risk_values.iloc[0])

    return coef_df, range_df, center, risk_cut

# ============================================================
# 辅助函数
# ============================================================
def get_display_name(variable, range_df):
    """优先使用 var_cp_all2_0.9.csv 中的 var_name2 展示变量名。"""
    matched = range_df.loc[
        range_df["Variable"].astype(str).str.strip().eq(variable)
    ]

    if not matched.empty:
        display_name = matched.iloc[0]["var_name2"]

        if pd.notna(display_name) and str(display_name).strip():
            return str(display_name).strip()

    fallback_names = {
        "Age": "Age",
        "Sex": "Sex",
        "marital_status": "Marital status",
        "SRH": "Self-rated health",
        "CMM_counts2": "Number of cardiometabolic conditions",
        "BMI": "BMI",
        "SP": "Systolic blood pressure",
        "DP": "Diastolic blood pressure",
        "hb": "Haemoglobin",
        "wbc": "White blood cell count",
        "plt": "Platelet count",
        "fbg": "Fasting blood glucose",
        "scr": "Serum creatinine",
        "hdl": "HDL cholesterol",
        "bun": "Blood urea nitrogen",
        "smoking_status3": "Smoking status",
        "ADL2": "ADL: Mild limitation",
        "ADL3": "ADL: Moderate limitation",
        "ADL4": "ADL: Complete limitation",
        "Education4": "Education: High/vocational",
        "Education5": "Education: College or above",
    }

    return fallback_names.get(variable, variable)

def get_range_row(variable, range_df):
    """获取某变量在范围文件中的一行。"""
    matched = range_df.loc[
        range_df["Variable"].astype(str).str.strip().eq(variable)
    ]

    if matched.empty:
        return None

    return matched.iloc[0]

def get_default_value(variable, range_df):
    """
    连续变量默认值：
    - 若有 Low 和 High，取二者中点；
    - 只有 Low，取 Low；
    - 只有 High，取 High；
    - 都没有则为 0。
    """
    row = get_range_row(variable, range_df)

    if row is None:
        return 0.0

    low = row["Low"]
    high = row["High"]

    has_low = pd.notna(low)
    has_high = pd.notna(high)

    if has_low and has_high:
        return float((float(low) + float(high)) / 2)

    if has_low:
        return float(low)

    if has_high:
        return float(high)

    return 0.0

# ============================================================
# 用户输入
# ============================================================
def get_user_inputs(coef_df, range_df):
    raw_values = {}

    # --------------------------------------------------------
    # 1. 人口学变量
    # 顺序：年龄、性别、教育程度、婚姻状况
    # --------------------------------------------------------
    st.subheader("Demographic characteristics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        raw_values["Age"] = float(
            st.number_input(
                "Age (years)",
                min_value=0,
                max_value=120,
                value=70,
                step=1,
                format="%d",
            )
        )

    with col2:
        sex = st.selectbox(
            "Sex",
            ["Female", "Male"],
        )
        raw_values["Sex"] = 0 if sex == "Female" else 1

    with col3:
        education = st.selectbox(
            "Education",
            [
                "Illiterate/semi-literate",
                "Primary school",
                "Middle school",
                "High/vocational school",
                "College or above",
            ],
        )

        raw_values["Education4"] = int(
            education == "High/vocational school"
        )
        raw_values["Education5"] = int(
            education == "College or above"
        )

    with col4:
        marital = st.selectbox(
            "Marital status",
            ["Partnered", "Unpartnered"],
        )
        raw_values["marital_status"] = (
            0 if marital == "Partnered" else 1
        )

    # --------------------------------------------------------
    # 2. 健康行为和功能状态
    # smoking history, ADL, self-rated health
    # --------------------------------------------------------
    st.subheader("Health behaviors and functional status")

    col1, col2, col3 = st.columns(3)

    with col1:
        smoking = st.selectbox(
            "Smoking history",
            ["Never-smoker", "Ex-smoker", "Current smoker"],
        )

        # 模型变量 smoking_status3：当前吸烟者为 1
        raw_values["smoking_status3"] = int(
            smoking == "Current smoker"
        )

    with col2:
        adl = st.selectbox(
            "Activities of daily living (ADL)",
            [
                "Independent",
                "Mild limitation",
                "Moderate limitation",
                "Complete limitation",
            ],
        )

        raw_values["ADL2"] = int(adl == "Mild limitation")
        raw_values["ADL3"] = int(adl == "Moderate limitation")
        raw_values["ADL4"] = int(adl == "Complete limitation")

    with col3:
        self_rated_health = st.selectbox(
            "Self-rated health",
            ["Optimal", "Suboptimal"],
        )

        raw_values["SRH"] = int(
            self_rated_health == "Suboptimal"
        )

    # --------------------------------------------------------
    # 3. 慢病情况
    # 用于计算 CMM_counts2
    # --------------------------------------------------------
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

    cmm_count = sum(
        [
            hypertension == "Yes",
            diabetes == "Yes",
            stroke == "Yes",
            heart_disease == "Yes",
        ]
    )

    raw_values["CMM_counts2"] = float(cmm_count)

    # --------------------------------------------------------
    # 4. 体格检查
    # --------------------------------------------------------
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

    with col3:
        bmi = weight / ((height / 100.0) ** 2)
        raw_values["BMI"] = float(bmi)

        st.metric(
            "Calculated BMI (kg/m²)",
            f"{bmi:.2f}",
        )

    # --------------------------------------------------------
    # 5. 血压和血液检测指标
    # SBP / DBP 显示整数；其余连续变量显示两位小数
    # --------------------------------------------------------
    st.subheader("Clinical and laboratory measurements")

    continuous_variables = [
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

    # 只展示 range 文件存在的变量；
    # DP、tc、tg、ldl 即使未进入最终 Cox 模型，也可保留为临床记录输入。
    available_variables = [
        variable
        for variable in continuous_variables
        if get_range_row(variable, range_df) is not None
    ]

    columns = st.columns(3)

    for index, variable in enumerate(available_variables):
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
# 风险计算
# ============================================================
def calculate_score(raw_values, coef_df, range_df, center):
    """
    模型计分逻辑：

    1. center 作为模型基线常数；
    2. 对连续变量：
       - 小于 Low 或大于 High，视为处于模型定义的危险范围；
       - 仅在危险范围时加入对应 Cox 系数；
    3. 对分类变量：
       - 当哑变量值为 1 时加入对应 Cox 系数；
    4. 仅保留实际产生贡献的指标，用于结果展示。
    """
    coefficient_map = dict(
        zip(coef_df["Variable"], coef_df["Coefficient"])
    )

    linear_score = float(center)
    contribution_rows = []

    for variable, coefficient in coefficient_map.items():
        coefficient = float(coefficient)
        value = float(raw_values.get(variable, 0))
        contribution = 0.0
        display_text = None

        range_row = get_range_row(variable, range_df)

        # ----------------------------------------------------
        # 连续变量：根据 Low / High 判断是否处于危险区间
        # ----------------------------------------------------
        if range_row is not None:
            low = range_row["Low"]
            high = range_row["High"]
            display_name = get_display_name(variable, range_df)

            low_is_valid = pd.notna(low)
            high_is_valid = pd.notna(high)

            if low_is_valid and value < float(low):
                contribution = coefficient
                display_text = f"{display_name} < {float(low):.2f}"

            elif high_is_valid and value > float(high):
                contribution = coefficient
                display_text = f"{display_name} > {float(high):.2f}"

        # ----------------------------------------------------
        # 分类变量：变量值为 1 时产生贡献
        # ----------------------------------------------------
        elif value == 1:
            contribution = coefficient
            display_text = get_display_name(variable, range_df)

        linear_score += contribution

        # 最终仅展示有贡献的指标
        if contribution != 0 and display_text is not None:
            contribution_rows.append(
                {
                    "Contributing indicator": display_text,
                    "Contribution": contribution,
                }
            )

    contribution_df = pd.DataFrame(contribution_rows)

    if not contribution_df.empty:
        contribution_df["Contribution"] = contribution_df[
            "Contribution"
        ].astype(float)

        contribution_df = contribution_df.sort_values(
            by="Contribution",
            key=lambda series: series.abs(),
            ascending=False,
        ).reset_index(drop=True)

        contribution_df["Contribution"] = contribution_df[
            "Contribution"
        ].map(lambda value: f"{value:+.2f}")

    return linear_score, contribution_df

# ============================================================
# 结果卡片
# ============================================================
def show_result_cards(score, risk_cut):
    high_risk = score >= risk_cut

    score_background = "#FEF2F2" if high_risk else "#EFF6FF"
    score_border = "#DC2626" if high_risk else "#2563EB"
    score_color = "#B91C1C" if high_risk else "#1D4ED8"

    risk_label = "High risk" if high_risk else "Low risk"
    risk_background = "#FEE2E2" if high_risk else "#DBEAFE"
    risk_border = "#EF4444" if high_risk else "#3B82F6"
    risk_color = "#B91C1C" if high_risk else "#1D4ED8"

    score_col, risk_col = st.columns(2)

    with score_col:
        st.markdown(
            f"""
            <div style="
                background-color: {score_background};
                border: 1px solid {score_border};
                border-radius: 14px;
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
                border-radius: 14px;
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
    st.caption(
        "RCS-LASSO-Cox model-based individual risk score calculator"
    )

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
            "Reference only: The calculated risk score and risk classification "
            "are for research and educational purposes only. They do not replace "
            "professional clinical assessment, diagnosis, or medical advice."
        )

        st.divider()
        st.subheader("Contributing risk indicators")

        if contribution_df.empty:
            st.info(
                "No model-defined contributing risk indicators were identified "
                "from the entered values."
            )
        else:
            st.dataframe(
                contribution_df,
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.caption(
        "Reference only: This calculator is intended for research and educational "
        "use. It must not be used as the sole basis for clinical decision-making."
    )

if __name__ == "__main__":
    main()