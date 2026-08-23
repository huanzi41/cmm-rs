from pathlib import Path
import io

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
# 文件读取
# ============================================================
def read_table(path_or_file):
    """自动读取真正的 CSV、Excel，以及扩展名错误的 Excel 文件。"""
    if hasattr(path_or_file, "read"):
        try:
            path_or_file.seek(0)
        except Exception:
            pass

        raw = path_or_file.read()
        name = getattr(path_or_file, "name", "")
        suffix = Path(name).suffix.lower()
        source = io.BytesIO(raw)
    else:
        path = Path(path_or_file)
        suffix = path.suffix.lower()
        raw = path.read_bytes()
        source = io.BytesIO(raw)

    # xlsx 文件本质上是 ZIP 文件，通常以 PK 开头。
    # 即使文件名错误地写成 .csv，也能正确识别。
    is_excel_binary = raw[:2] == b"PK"

    if suffix in {".xlsx", ".xls"} or is_excel_binary:
        source.seek(0)
        return pd.read_excel(source, engine="openpyxl")

    # 先按 CSV 读取；如果内容实际是 Excel 或格式异常，再尝试 Excel。
    try:
        source.seek(0)
        df = pd.read_csv(source)

        # 如果 CSV 只读出了一个异常列，仍然保留结果。
        return df
    except Exception:
        source.seek(0)
        return pd.read_excel(source, engine="openpyxl")

def clean_columns(df):
    """清理列名，避免列名前后空格造成匹配失败。"""
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    return df

# ============================================================
# risk_cut 读取
# ============================================================
def extract_risk_cut(cut_df):
    """
    兼容以下 risk_cut 文件格式：

    格式一：
        Model,Risk_cut
        RCS_lasso_cox,1.1065112042101

    格式二：
        RCS_lasso_cox,1.1065112042101

    格式三：
        1.1065112042101

    格式四：Excel 中只有一个数字。
    """
    cut_df = cut_df.copy()

    # 情况一：标准格式，第一列是模型名称，第二列是阈值
    model_column = None
    for column in cut_df.columns:
        if str(column).strip().lower() in {
            "model",
            "model_name",
            "modelname",
            "method",
        }:
            model_column = column
            break

    if model_column is not None:
        model_values = (
            cut_df[model_column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        matched = cut_df.loc[
            model_values.isin({"rcs_lasso_cox", "rcs-lasso-cox"})
        ]

        if not matched.empty:
            for column in matched.columns:
                if column == model_column:
                    continue
                value = pd.to_numeric(matched.iloc[0][column], errors="coerce")
                if pd.notna(value):
                    return float(value)

    # 情况二：第一列直接是模型名，第二列是数字
    if cut_df.shape[1] >= 2:
        first_column = cut_df.iloc[:, 0].astype(str).str.strip().str.lower()
        matched = cut_df.loc[
            first_column.isin({"rcs_lasso_cox", "rcs-lasso-cox"})
        ]

        if not matched.empty:
            for column in cut_df.columns[1:]:
                value = pd.to_numeric(matched.iloc[0][column], errors="coerce")
                if pd.notna(value):
                    return float(value)

    # 情况三：文件中只有一个或多个数字，直接取第一个数字
    numeric_values = pd.to_numeric(cut_df.stack(), errors="coerce").dropna()

    if not numeric_values.empty:
        return float(numeric_values.iloc[0])

    raise ValueError(
        "risk_cut.csv 中没有找到有效的风险阈值。请填写类似 1.1065112042101 的数字。"
    )

# ============================================================
# 模型读取
# ============================================================
def load_model(uploaded_coef=None, uploaded_range=None, uploaded_cut=None):
    coef_source = uploaded_coef if uploaded_coef is not None else COEF_FILE
    range_source = uploaded_range if uploaded_range is not None else RANGE_FILE
    cut_source = uploaded_cut if uploaded_cut is not None else CUT_FILE

    coef_df = clean_columns(read_table(coef_source))
    range_df = clean_columns(read_table(range_source))
    cut_df = clean_columns(read_table(cut_source))

    if coef_df.shape[1] < 2:
        raise ValueError("cox_coefficients.csv 至少需要两列：变量名和系数。")

    if range_df.shape[1] < 3:
        raise ValueError("var_cp_all2_0.9.csv 至少需要三列：变量名、Low、High。")

    # 只依赖列的位置，不依赖原始列名。
    coef_df = coef_df.iloc[:, :2].copy()
    coef_df.columns = ["Variable", "Coefficient"]

    range_df = range_df.iloc[:, :3].copy()
    range_df.columns = ["Variable", "Low", "High"]

    coef_df["Variable"] = coef_df["Variable"].astype(str).str.strip()
    coef_df["Coefficient"] = pd.to_numeric(
        coef_df["Coefficient"], errors="coerce"
    )

    range_df["Variable"] = range_df["Variable"].astype(str).str.strip()
    range_df["Low"] = pd.to_numeric(range_df["Low"], errors="coerce")
    range_df["High"] = pd.to_numeric(range_df["High"], errors="coerce")

    coef_df = coef_df.dropna(subset=["Variable", "Coefficient"])
    range_df = range_df.dropna(subset=["Variable", "Low", "High"])

    center_rows = coef_df.loc[
        coef_df["Variable"].str.lower() == "center", "Coefficient"
    ]

    if center_rows.empty:
        raise ValueError(
            "cox_coefficients.csv 中必须包含一个名为 center 的变量。"
        )

    center = float(center_rows.iloc[0])
    risk_cut = extract_risk_cut(cut_df)

    # center 是模型校准常数，不作为普通变量再次计算。
    coef_df = coef_df[
        coef_df["Variable"].str.lower() != "center"
    ].copy()

    return coef_df, range_df, center, risk_cut

# ============================================================
# 用户输入
# ============================================================
def get_user_inputs(coef_df, range_df):
    raw_values = {}

    st.subheader("Basic Demographics")
    cols = st.columns(3)

    with cols[0]:
        sex = st.selectbox("Sex", ["Female", "Male"])
        raw_values["Sex"] = 0 if sex == "Female" else 1

    with cols[1]:
        marital = st.selectbox("Marital status", ["Partnered", "Unpartnered"])
        raw_values["marital_status"] = 0 if marital == "Partnered" else 1

    with cols[2]:
        raw_values["Age"] = float(
            st.number_input("Age", min_value=0, max_value=120, value=70, step=1)
        )

    with cols[0]:
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

    with cols[1]:
        smoking = st.selectbox(
            "Smoking status",
            ["Never-smoker", "Ex-smoker", "Smoker"],
        )
        raw_values["smoking_status3"] = int(smoking == "Smoker")

    with cols[2]:
        adl = st.selectbox("ADL", ["Independent", "Mild", "Moderate", "Complete"])
        raw_values["ADL2"] = int(adl == "Mild")
        raw_values["ADL3"] = int(adl == "Moderate")
        raw_values["ADL4"] = int(adl == "Complete")

    st.subheader("Self-rated Health & Chronic Conditions")
    cols = st.columns(3)

    with cols[0]:
        health = st.selectbox("Self-rated health", ["Optimal", "Suboptimal"])
        raw_values["SRH"] = int(health == "Suboptimal")

    with cols[1]:
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        raw_values["Hypertension"] = int(hypertension == "Yes")

    with cols[2]:
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        raw_values["Diabetes"] = int(diabetes == "Yes")

    cols = st.columns(3)

    with cols[0]:
        stroke = st.selectbox("Stroke", ["No", "Yes"])
        raw_values["Stroke"] = int(stroke == "Yes")

    with cols[1]:
        heart_disease = st.selectbox("Heart disease", ["No", "Yes"])
        raw_values["Heart disease"] = int(heart_disease == "Yes")

    st.subheader("Body Measurement")
    cols = st.columns(3)

    with cols[0]:
        height = st.number_input(
            "Height (cm)",
            min_value=50.0,
            max_value=250.0,
            value=165.0,
            step=1.0,
        )

    with cols[1]:
        weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=250.0,
            value=65.0,
            step=0.5,
        )

    bmi = weight / ((height / 100.0) ** 2)
    raw_values["BMI"] = round(bmi, 4)
    st.info(f"Calculated BMI = **{bmi:.2f}** kg/m²")

    st.subheader("Continuous Clinical Variables")
    continuous_variables = [
        "SP", "DP", "hb", "wbc", "plt", "fbg",
        "scr", "tc", "tg", "ldl", "hdl", "bun",
    ]

    cols = st.columns(3)

    for index, variable in enumerate(continuous_variables):
        row = range_df.loc[range_df["Variable"] == variable]

        if row.empty:
            default_value = 0.0
        else:
            low = float(row.iloc[0]["Low"])
            high = float(row.iloc[0]["High"])
            default_value = (low + high) / 2.0

        raw_values[variable] = cols[index % 3].number_input(
            variable,
            value=float(default_value),
            format="%.4f",
        )

    return raw_values

# ============================================================
# 风险计算
# ============================================================
def calculate_score(raw_values, coef_df, range_df, center):
    coefficients = coef_df.set_index("Variable")["Coefficient"].to_dict()
    ranges = range_df.set_index("Variable")[["Low", "High"]].to_dict("index")

    linear_score = float(center)
    rows = []

    for variable, coefficient in coefficients.items():
        if variable not in raw_values:
            value = 0.0
        else:
            value = float(raw_values[variable])

        contribution = value * float(coefficient)
        linear_score += contribution

        rows.append(
            {
                "Variable": variable,
                "Value": value,
                "Coefficient": float(coefficient),
                "Contribution": contribution,
            }
        )

    detail_df = pd.DataFrame(rows)
    return linear_score, detail_df

# ============================================================
# 主程序
# ============================================================
def main():
    st.title("Cardiometabolic Multimorbidity Risk Prediction")
    st.caption("Cox model-based individual risk score calculator")

    with st.sidebar:
        st.header("Model files")
        uploaded_coef = st.file_uploader(
            "cox_coefficients.csv / xlsx",
            type=["csv", "xlsx", "xls"],
        )
        uploaded_range = st.file_uploader(
            "var_cp_all2_0.9.csv / xlsx",
            type=["csv", "xlsx", "xls"],
        )
        uploaded_cut = st.file_uploader(
            "risk_cut.csv / xlsx",
            type=["csv", "xlsx", "xls"],
        )

    try:
        coef_df, range_df, center, risk_cut = load_model(
            uploaded_coef,
            uploaded_range,
            uploaded_cut,
        )
    except Exception as error:
        st.error(f"Unable to load model files: {error}")
        st.stop()

    st.success("Model files loaded successfully.")

    raw_values = get_user_inputs(coef_df, range_df)

    if st.button("Calculate risk", type="primary"):
        score, detail_df = calculate_score(
            raw_values,
            coef_df,
            range_df,
            center,
        )

        st.subheader("Prediction result")
        st.metric("Risk score", f"{score:.6f}")
        st.metric("Risk cut-off", f"{risk_cut:.6f}")

        if score >= risk_cut:
            st.error("Predicted risk: High")
        else:
            st.success("Predicted risk: Low")

        with st.expander("Calculation details"):
            st.dataframe(detail_df, use_container_width=True)

if __name__ == "__main__":
    main()