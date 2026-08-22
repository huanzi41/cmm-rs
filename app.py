import streamlit as st

# =========================
# 页面基础配置
# =========================
st.set_page_config(
    page_title="Cardiovascular Disease Risk Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================
# 页面样式
# =========================
st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 42px;
        padding-bottom: 50px;
    }

    h1 {
        color: #30323d;
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    h2 {
        color: #30323d;
        font-size: 25px;
        font-weight: 650;
        margin-top: 30px;
        margin-bottom: 18px;
    }

    .subtitle {
        color: #6b6f7b;
        font-size: 15px;
        margin-bottom: 30px;
    }

    label {
        color: #4b4f5a !important;
        font-size: 13px !important;
    }

    div[data-baseweb="input"] {
        background-color: #f1f2f6;
        border-radius: 6px;
        border: 1px solid transparent;
    }

    div[data-baseweb="input"]:focus-within {
        border: 1px solid #ff6b6b;
    }

    input {
        color: #30323d !important;
    }

    .stButton > button {
        color: #ff5c5c;
        background-color: #ffffff;
        border: 1px solid #ff5c5c;
        border-radius: 6px;
        padding: 8px 22px;
        font-weight: 600;
    }

    .stButton > button:hover {
        color: #ffffff;
        background-color: #ff5c5c;
        border-color: #ff5c5c;
    }

    .result-box {
        border: 1px solid #e5e6eb;
        border-radius: 8px;
        padding: 24px;
        margin-top: 12px;
        background-color: #fafafa;
    }

    .result-label {
        color: #6b6f7b;
        font-size: 14px;
        margin-bottom: 5px;
    }

    .result-value {
        color: #30323d;
        font-size: 32px;
        font-weight: 700;
    }

    .risk-low {
        color: #1c8c5e;
        font-size: 24px;
        font-weight: 700;
    }

    .risk-medium {
        color: #d68a00;
        font-size: 24px;
        font-weight: 700;
    }

    .risk-high {
        color: #d64545;
        font-size: 24px;
        font-weight: 700;
    }

    .warning-title {
        color: #b42318;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .warning-item {
        color: #6b2a25;
        font-size: 14px;
        padding: 3px 0;
    }

    .footer-note {
        color: #8a8e99;
        font-size: 12px;
        margin-top: 35px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# 标题
# =========================
st.markdown(
    """
    <h1>Cardiovascular Disease Risk Prediction</h1>
    <div class="subtitle">
        Calculate the estimated cardiovascular disease risk based on patient clinical data.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h2>Enter Patient Data</h2>", unsafe_allow_html=True)

# =========================
# 输入区域
# =========================
row1 = st.columns(3)

with row1[0]:
    age = st.number_input(
        "Age (years)",
        min_value=0.0,
        max_value=150.0,
        value=50.0,
        step=1.0,
    )

with row1[1]:
    gender = st.selectbox(
        "Sex",
        options=["Male", "Female"],
    )

with row1[2]:
    tc = st.number_input(
        "TC (mmol/L)",
        min_value=0.0,
        max_value=1000.0,
        value=5.00,
        step=0.01,
        format="%.2f",
    )

row2 = st.columns(3)

with row2[0]:
    sbp = st.number_input(
        "SBP (mmHg)",
        min_value=0.0,
        max_value=1000.0,
        value=120.0,
        step=0.1,
        format="%.2f",
    )

with row2[1]:
    dbp = st.number_input(
        "DBP (mmHg)",
        min_value=0.0,
        max_value=500.0,
        value=80.0,
        step=0.1,
        format="%.2f",
    )

with row2[2]:
    bmi = st.number_input(
        "BMI (kg/m²)",
        min_value=0.0,
        max_value=200.0,
        value=24.0,
        step=0.01,
        format="%.2f",
    )

row3 = st.columns(3)

with row3[0]:
    hb = st.number_input(
        "Hb (g/dL)",
        min_value=0.0,
        max_value=100.0,
        value=14.0,
        step=0.01,
        format="%.2f",
    )

with row3[1]:
    wbc = st.number_input(
        "WBC (10⁹/L)",
        min_value=0.0,
        max_value=1000.0,
        value=6.0,
        step=0.01,
        format="%.2f",
    )

with row3[2]:
    plt = st.number_input(
        "Plt (10⁹/L)",
        min_value=0.0,
        max_value=5000.0,
        value=250.0,
        step=0.01,
        format="%.2f",
    )

row4 = st.columns(3)

with row4[0]:
    fbg = st.number_input(
        "FBG (mmol/L)",
        min_value=0.0,
        max_value=500.0,
        value=5.5,
        step=0.01,
        format="%.2f",
    )

with row4[1]:
    scr = st.number_input(
        "Scr (μmol/L)",
        min_value=0.0,
        max_value=10000.0,
        value=80.0,
        step=0.01,
        format="%.2f",
    )

with row4[2]:
    tg = st.number_input(
        "TG (mmol/L)",
        min_value=0.0,
        max_value=500.0,
        value=1.50,
        step=0.01,
        format="%.2f",
    )

row5 = st.columns(3)

with row5[0]:
    ldl_c = st.number_input(
        "LDL-C (mmol/L)",
        min_value=0.0,
        max_value=500.0,
        value=3.00,
        step=0.01,
        format="%.2f",
    )

with row5[1]:
    hdl_c = st.number_input(
        "HDL-C (mmol/L)",
        min_value=0.0,
        max_value=500.0,
        value=1.30,
        step=0.01,
        format="%.2f",
    )

with row5[2]:
    bun = st.number_input(
        "BUN (mmol/L)",
        min_value=0.0,
        max_value=500.0,
        value=5.00,
        step=0.01,
        format="%.2f",
    )

st.write("")

# =========================
# 计算按钮
# =========================
calculate = st.button("Calculate Risk", type="secondary")

if calculate:
    gender_score = 0.02 if gender == "Male" else 0.01

    # 按用户提供的公式计算。
    # TC 在原始公式中出现两次，因此两个 TC 项均保留。
    total_score = (
        0.01 * age
        + gender_score
        + 0.03 * tc
        + 0.004 * sbp
        + 0.003 * dbp
        + 0.007 * bmi
        + 0.005 * hb
        + 0.009 * wbc
        + 0.004 * plt
        + 0.0002 * fbg
        + 0.006 * scr
        + 0.009 * tc
        + 0.004 * tg
        + 0.005 * ldl_c
        + 0.008 * hdl_c
        + 0.02 * bun
    )

    if total_score < 1:
        risk_probability = 30
        risk_class = "risk-low"
        risk_text = "Low Risk"
    elif total_score <= 2:
        risk_probability = 60
        risk_class = "risk-medium"
        risk_text = "Moderate Risk"
    else:
        risk_probability = 90
        risk_class = "risk-high"
        risk_text = "High Risk"

    st.markdown("<h2>Prediction Result</h2>", unsafe_allow_html=True)

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">Total Score</div>
                <div class="result-value">{total_score:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with result_col2:
        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">Estimated Risk Probability</div>
                <div class="{risk_class}">{risk_probability}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with result_col3:
        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">Risk Category</div>
                <div class="{risk_class}">{risk_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================
    # 高风险指标检查
    # =========================
    warnings = []

    if sbp < 114.13 or sbp > 171.8:
        warnings.append(f"SBP = {sbp:.2f} mmHg，处于高风险范围：<114.13 或 >171.8")

    if dbp < 62.95 or dbp > 97.98:
        warnings.append(f"DBP = {dbp:.2f} mmHg，处于高风险范围：<62.95 或 >97.98")

    if bmi < 20.84:
        warnings.append(f"BMI = {bmi:.2f} kg/m²，处于高风险范围：<20.84")

    if hb < 11.91:
        warnings.append(f"Hb = {hb:.2f} g/dL，处于高风险范围：<11.91")

    if wbc < 3.55 or wbc > 8.97:
        warnings.append(f"WBC = {wbc:.2f} ×10⁹/L，处于高风险范围：<3.55 或 >8.97")

    if plt < 125.05 or plt > 316.42:
        warnings.append(f"Plt = {plt:.2f} ×10⁹/L，处于高风险范围：<125.05 或 >316.42")

    if fbg < 3.88 or fbg > 10.44:
        warnings.append(f"FBG = {fbg:.2f} mmol/L，处于高风险范围：<3.88 或 >10.44")

    if scr < 34.68 or scr > 107.65:
        warnings.append(f"Scr = {scr:.2f} μmol/L，处于高风险范围：<34.68 或 >107.65")

    if tc < 3.18 or tc > 7.12:
        warnings.append(f"TC = {tc:.2f} mmol/L，处于高风险范围：<3.18 或 >7.12")

    if tg < 0.66 or tg > 3.54:
        warnings.append(f"TG = {tg:.2f} mmol/L，处于高风险范围：<0.66 或 >3.54")

    if ldl_c < 1.46 or ldl_c > 4.74:
        warnings.append(f"LDL-C = {ldl_c:.2f} mmol/L，处于高风险范围：<1.46 或 >4.74")

    if hdl_c < 0.92 or hdl_c > 3.57:
        warnings.append(f"HDL-C = {hdl_c:.2f} mmol/L，处于高风险范围：<0.92 或 >3.57")

    if bun < 2.02 or bun > 8.08:
        warnings.append(f"BUN = {bun:.2f} mmol/L，处于高风险范围：<2.02 或 >8.08")

    st.markdown("<h2>High-risk Indicators</h2>", unsafe_allow_html=True)

    if warnings:
        warning_html = '<div class="result-box">'
        warning_html += '<div class="warning-title">检测到以下指标处于高风险范围：</div>'

        for warning in warnings:
            warning_html += f'<div class="warning-item">• {warning}</div>'

        warning_html += "</div>"
        st.markdown(warning_html, unsafe_allow_html=True)
    else:
        st.success("目前没有检测到处于指定高风险范围内的指标。")

st.markdown(
    """
    <div class="footer-note">
        本工具仅用于风险估算和辅助筛查，不能替代医生诊断、实验室复核或临床决策。
        风险概率根据预设分段规则计算，不代表经过临床验证的个体化发病概率。
    </div>
    """,
    unsafe_allow_html=True,
)