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

# 注意：虽然文件扩展名是 .csv，但实际内容是 Excel 文件
COEF_FILE = BASE_DIR / "cox_coefficients.csv"
RANGE_FILE = BASE_DIR / "var_cp_all2_0.9.csv"
CUT_FILE = BASE_DIR / "risk_cut.csv"

# ============================================================
# 内置文件读取
# ============================================================
@st.cache_data
def load_model():
    """
    读取内置模型文件。

    文件实际格式为 Excel 工作簿，即使扩展名为 .csv，
    也统一使用 pd.read_excel(..., engine="openpyxl")。
    """
    try:
        coef_df = pd.read_excel(COEF_FILE, engine="openpyxl")
        range_df = pd.read_excel(RANGE_FILE, engine="openpyxl")
        cut_df = pd.read_excel(CUT_FILE, engine="openpyxl")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Model file not found: {error.filename}. "
            "Please make sure all three built-in files are in the same folder as app.py."
        ) from error

    coef_df = coef_df.copy()
    range_df = range_df.copy()
    cut_df = cut_df.copy()

    coef_df.columns = [str(col).strip() for col in coef_df.columns]
    range_df.columns = [str(col).strip() for col in range_df.columns]
    cut_df.columns = [str(col).strip() for col in cut_df.columns]

    # --------------------------------------------------------
    # Cox 系数表
    # 默认前两列：变量名、系数
    # --------------------------------------------------------
    if coef_df.shape[1] < 2:
        raise ValueError(
            "cox_coefficients.csv 至少需要两列：变量名和 Cox coefficient。"
        )

    coef_df = coef_df.iloc[:, :2].copy()
    coef_df.columns = ["Variable", "Coefficient"]

    coef_df["Variable"] = coef_df["Variable"].astype(str).str.strip()
    coef_df["Coefficient"] = pd.to_numeric(
        coef_df["Coefficient"],
        errors="coerce",
    )

    coef_df = coef_df.dropna(subset=["Variable", "Coefficient"])

    # 获取模型中心常数
    center_rows = coef_df.loc[
        coef_df["Variable"].str.lower().eq("center"),
        "Coefficient",
    ]

    if center_rows.empty:
        raise ValueError(
            "cox_coefficients.csv 中必须包含名为 center 的模型常数。"
        )

    center = float(center_rows.iloc[0])

    # center 不作为普通预测变量
    coef_df = coef_df.loc[
        ~coef_df["Variable"].str.lower().eq("center")
    ].copy()

    # --------------------------------------------------------
    # 连续变量临界值表
    # 至少应包含 Variable、Low、High；
    # var_name2 用于页面和结果的变量展示名称
    # --------------------------------------------------------
    if range_df.shape[1] < 3:
        raise ValueError(
            "var_cp_all2_0.9.csv 至少需要三列：Variable、Low、High。"
        )

    # 若没有规范列名，则默认前 3 列依次为 Variable、Low、High
    if "Variable" not in range_df.columns:
        range_df = range_df.rename(
            columns={range_df.columns[0]: "Variable"}
        )

    if "Low" not in range_df.columns:
        remaining_columns = [
            col for col in range_df.columns if col != "Variable"
        ]
        range_df = range_df.rename(
            columns={remaining_columns[0]: "Low"}
        )

    if "High" not in range_df.columns:
        remaining_columns = [
            col
            for col in range_df.columns
            if col not in {"Variable", "Low"}
        ]
        range_df = range_df.rename(
            columns={remaining_columns[0]: "High"}
        )

    # 如果文件没有 var_name2，默认以 Variable 作为展示名称
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
    # 风险分层阈值
    # 从 risk_cut 文件中获取第一个有效数字
    # --------------------------------------------------------
    numeric_values = pd.to_numeric(
        cut_df.stack(),
        errors="coerce",
    ).dropna()

    if numeric_values.empty:
        raise ValueError(
            "risk_cut.csv 中没有找到有效的风险分层阈值。"
        )

    risk_cut = float(numeric_values.iloc[0])

    return coef_df, range_df, center, risk_cut

# ============================================================
# 辅助函数
# ============================================================
def get_range_row(variable, range_df):
    """获取某变量对应的 Low、High、var_name2 信息。"""
    matched = range_df.loc[
        range_df["Variable"].astype(str).str.strip().eq(variable)
    ]

    if matched.empty:
        return None

    return matched.iloc[0]

def get_display_name(variable, range_df):
    """优先使用 var_cp_all2_0.9.csv 中的 var_name2 作为展示名称。"""
    range_row = get_range_row(variable, range_df)

    if range_row is not None:
        name = range_row.get("var_name2", variable)

        if pd.notna(name) and str(name).strip():
            return str(name).strip()

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
        "smoking_status3": "Smoking history",
        "ADL2": "ADL: Mild",
        "ADL3": "ADL: Moderate",
        "ADL4": "ADL: Complete",
        "Education4": "Education: High/vocational",
        "Education5": "Education: College or above",
    }

    return fallback_names.get(variable, variable)

def get_default_value(variable, range_df):
    """
    连续变量的默认初始值：
    - 同时有 Low、High：取中点；
    - 仅有 Low：使用 Low；
    - 仅有 High：使用 High；
    - 无范围数据：使用 0。
    """
    row = get_range_row(variable, range_df)

    if row is None:
        return 0.0

    low = row.get("Low")
    high = row.get("High")

    if pd.notna(low) and pd.notna(high):
        return float((float(low) + float(high)) / 2)

    if pd.notna(low):
        return float(low)

    if pd.notna(high):
        return float(high)

    return 0.0

def model_contains_variable(variable, coef_df):
    """判断变量是否进入 Cox 模型。"""
    return variable in set(coef_df["Variable"].astype(str))

# ============================================================
# 用户输入
# ============================================================
def get_user_inputs(coef_df, range_df):
    raw_values = {}

    # --------------------------------------------------------
    # 人口学变量
    # 年龄、性别、教育程度、婚姻状况
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
        raw_values["marital_status"] = (
            0 if marital == "Partnered" else 1
        )

    # --------------------------------------------------------
    # 健康行为和功能状态
    # smoking history, ADL, self-rated health
    # --------------------------------------------------------
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
        # 根据要求：仅显示 Independent、Mild、Moderate、Complete
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

    # --------------------------------------------------------
    # 心代谢慢病情况
    # 用于得到 CMM_counts2
    # --------------------------------------------------------
    st.subheader("Cardiometabolic conditions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        hypertension = st.selectbox(
            "Hypertension",
            ["No", "Yes"],
        )

    with col2:
        diabetes = st.selectbox(
            "Diabetes",
            ["No", "Yes"],
        )

    with col3:
        stroke = st.selectbox(
            "Stroke",
            ["No", "Yes"],
        )

    with col4:
        heart_disease = st.selectbox(
            "Heart disease",
            ["No", "Yes"],
        )

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
    # 体格检查：身高、体重和 BMI
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

    bmi = weight / ((height / 100.0) ** 2)
    raw_values["BMI"] = float(bmi)

    with col3:
        st.metric(
            "Calculated BMI (kg/m²)",
            f"{bmi:.2f}",
        )

    # --------------------------------------------------------
    # 血压及血检指标
    # SP、DP 使用整数；其余连续变量使用两位小数
    # --------------------------------------------------------
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

    # 只有变量存在于范围表或 Cox 表时才显示
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
            # 收缩压、舒张压默认显示为整数
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

            # 其他连续变量默认显示两位小数
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
    评分逻辑：

    1. center 为模型基线常数；
    2. 连续变量：
       若数值低于 Low 或高于 High，则计入该变量的 Cox coefficient；
    3. 分类变量：
       若对应哑变量为 1，则计入 Cox coefficient；
    4. 最终只保留产生实际贡献的危险指标。
    """
    coefficient_map = dict(
        zip(
            coef_df["Variable"].astype(str),
            coef_df["Coefficient"].astype(float),
        )
    )

    score = float(center)
    contribution_rows = []

    for variable, coefficient in coefficient_map.items():
        value = float(raw_values.get(variable, 0))
        contribution = 0.0
        indicator_text = None

        range_row = get_range_row(variable, range_df)

        # ----------------------------------------------------
        # 连续变量：根据 Low / High 判断是否进入危险范围
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # 分类变量：值为 1 时计入系数
        # ----------------------------------------------------
        elif value == 1:
            contribution = coefficient
            indicator_text = get_display_name(variable, range_df)

        score += contribution

        # 只保存实际贡献指标
        if contribution != 0 and indicator_text is not None:
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
            key=lambda series: series.abs(),
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
    """
    只显示：
    - Risk score
    - Risk cut-off（显示 High risk / Low risk，而非具体阈值）
    """
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
            "Reference only: The calculated risk score and risk classification "
            "are for research and educational purposes only and do not replace "
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
        "use and must not be used as the sole basis for clinical decision-making."
    )

if __name__ == "__main__":
    main()