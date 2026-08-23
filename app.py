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

def read_table(path_or_file):
    """Read CSV or Excel files, including uploaded files."""
    if hasattr(path_or_file, "read"):
        raw = path_or_file.read()
        name = getattr(path_or_file, "name", "")
        suffix = Path(name).suffix.lower()
        buffer = io.BytesIO(raw)
    else:
        suffix = Path(path_or_file).suffix.lower()
        buffer = path_or_file

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(buffer)

    try:
        if hasattr(buffer, "seek"):
            buffer.seek(0)
        return pd.read_csv(buffer)
    except Exception:
        if hasattr(buffer, "seek"):
            buffer.seek(0)
        return pd.read_excel(buffer)

def load_model(uploaded_coef, uploaded_range, uploaded_cut):
    coef_df = read_table(uploaded_coef if uploaded_coef else COEF_FILE)
    range_df = read_table(uploaded_range if uploaded_range else RANGE_FILE)
    cut_df = read_table(uploaded_cut if uploaded_cut else CUT_FILE)

    coef_df.columns = [str(c).strip() for c in coef_df.columns]
    range_df.columns = [str(c).strip() for c in range_df.columns]
    cut_df.columns = [str(c).strip() for c in cut_df.columns]

    coef_df = coef_df.rename(
        columns={
            coef_df.columns[0]: "Variable",
            coef_df.columns[1]: "Coefficient",
        }
    )
    range_df = range_df.rename(
        columns={
            range_df.columns[0]: "Variable",
            range_df.columns[1]: "Low",
            range_df.columns[2]: "High",
        }
    )

    coef_df["Variable"] = coef_df["Variable"].astype(str).str.strip()
    coef_df["Coefficient"] = pd.to_numeric(coef_df["Coefficient"], errors="coerce")
    range_df["Variable"] = range_df["Variable"].astype(str).str.strip()
    range_df["Low"] = pd.to_numeric(range_df["Low"], errors="coerce")
    range_df["High"] = pd.to_numeric(range_df["High"], errors="coerce")

    coef_df = coef_df.dropna(subset=["Variable", "Coefficient"])
    range_df = range_df.dropna(subset=["Variable", "Low", "High"])

    center_rows = coef_df.loc[coef_df["Variable"].str.lower() == "center", "Coefficient"]
    if center_rows.empty:
        raise ValueError("cox_coefficients.csv must contain a row named 'center'.")
    center = float(center_rows.iloc[0])

    cut_rows = cut_df.loc[
        cut_df.iloc[:, 0].astype(str).str.strip().str.lower() == "rcs_lasso_cox",
        cut_df.iloc[:, 1],
    ]
    if cut_rows.empty:
        raise ValueError("risk_cut.csv must contain the model RCS_lasso_cox.")
    risk_cut = float(pd.to_numeric(cut_rows.iloc[0]))

    coef_df = coef_df[coef_df["Variable"].str.lower() != "center"].copy()
    return coef_df, range_df, center, risk_cut

def get_user_inputs(coef_df, range_df):
    raw_values = {}

    st.subheader("Basic Demographics")
    cols = st.columns(3)

    with cols[0]:
        sex_label = st.selectbox("Sex", ["Female", "Male"])
        raw_values["Sex"] = 0 if sex_label == "Female" else 1

    with cols[1]:
        marital_label = st.selectbox("Marital status", ["Partnered", "Unpartnered"])
        raw_values["marital_status"] = 0 if marital_label == "Partnered" else 1

    with cols[2]:
        age = st.number_input("Age", min_value=0, max_value=120, value=70, step=1)
        raw_values["Age"] = float(age)

    with cols[0]:
        edu_label = st.selectbox(
            "Education",
            [
                "Illiterate/semi-literate",
                "Primary",
                "Middle",
                "High/vocational",
                "College or above",
            ],
        )
        raw_values["Education4"] = 1 if edu_label == "High/vocational" else 0
        raw_values["Education5"] = 1 if edu_label == "College or above" else 0

    with cols[1]:
        smoking_label = st.selectbox(
            "Smoking status", ["Never-smoker", "Ex-smoker", "Smoker"]
        )
        raw_values["smoking_status3"] = 1 if smoking_label == "Smoker" else 0

    with cols[2]:
        adl_label = st.selectbox("ADL", ["Independent", "Mild", "Moderate", "Complete"])
        raw_values["ADL2"] = 1 if adl_label == "Mild" else 0
        raw_values["ADL3"] = 1 if adl_label == "Moderate" else 0
        raw_values["ADL4"] = 1 if adl_label == "Complete" else 0

    st.subheader("Self-rated Health & Chronic Conditions")
    cols = st.columns(3)

    with cols[0]:
        srh_label = st.selectbox("Self-rated health", ["Optimal", "Suboptimal"])
        raw_values["SRH"] = 0 if srh_label == "Optimal" else 1

    with cols[1]:
        htn_label = st.selectbox("Hypertension", ["No", "Yes"])
        raw_values["Hypertension"] = 1 if htn_label == "Yes" else 0

    with cols[2]:
        dm_label = st.selectbox("Diabetes", ["No", "Yes"])
        raw_values["Diabetes"] = 1 if dm_label == "Yes" else 0

    cols = st.columns(2 + 1)
    with cols[0]:
        stroke_label = st.selectbox("Stroke", ["No", "Yes"])
        raw_values["Stroke"] = 1 if stroke_label == "Yes" else 0

    with cols[1]:
        chd_label = st.selectbox("Heart disease", ["No", "Yes"])
        raw_values["Heart disease"] = 1 if chd_label == "Yes" else 0

    st.subheader("Body Measurement")
    cols = st.columns(3)
    with cols[0]:
        height_cm = st.number_input(
            "Height (cm)", min_value=50.0, max_value=250.0, value=165.0, step=1.0
        )
    with cols[1]:
        weight_kg = st.number_input(
            "Weight (kg)", min_value=20.0, max_value=250.0, value=65.0, step=0.5
        )
    bmi = round(weight_kg / ((height_cm / 100.0) ** 2), 4) if height_cm > 0 else 0.0
    st.info(f"Calculated BMI = **{bmi:.2f}** kg/m²")
    raw_values["BMI"] = bmi

    st.subheader("Continuous Clinical Variables")
    continuous_order = ["SP", "DP", "hb", "wbc", "plt", "fbg", "scr", "tc", "tg", "ldl", "hdl", "bun"]
    cols = st.columns(3)
    for idx, var in enumerate(continuous_order):
        col = cols[idx % 3]
        row = range_df.loc[range_df["Variable"] == var]
        if not row.empty:
            mid = float((row.iloc[0]["Low"] + row.iloc[0]["High"]) / 2)
        else:
            mid = 5.0
        raw_values[var] = col.number_input(
            f"{var}", value=mid, format="%.4f"
        )

    return raw_values

def calculate_score(raw_values, coef_df, range_df, center):
    ranges = range_df.set_index("Variable")[["Low", "High"]].to_dict("index")
    coefficients = coef_df.set_index("Variable")["Coefficient"].to_dict()

    rows = []
    linear_score = 0.0

    for variable, coefficient in coefficients.items():
        value = raw_values.get(variable)
        if value is None:
            continue

        if variable in ranges:
            low = ranges[variable]["Low"]
            high = ranges[variable]["High"]
            indicator = int(value < low or value > high)
            rule = f"1 if value < {low:g} or value > {high:g}; otherwise 0"
        else:
            indicator = float(value)
            low = None
            high = None
            rule = "Categorical / pre-encoded indicator"

        contribution = indicator * coefficient
        linear_score += contribution

        rows.append(
            {
                "Variable": variable,
                "Observed value": value,
                "Indicator": indicator,
                "Coefficient": coefficient,
                "Contribution": contribution,
                "Rule": rule,
                "Active": indicator != 0,
            }
        )

    adjusted_score = linear_score - center
    return adjusted_score, pd.DataFrame(rows)

# -----------------------------
# Sidebar & Run
# -----------------------------
st.sidebar.title("Model Configuration")
st.sidebar.caption("Optional file replacement")

uploaded_coef = st.sidebar.file_uploader(
    "Cox coefficients", type=["csv", "xlsx", "xls"]
)
uploaded_range = st.sidebar.file_uploader(
    "Continuous-variable ranges", type=["csv", "xlsx", "xls"]
)
uploaded_cut = st.sidebar.file_uploader(
    "Risk cut-off", type=["csv", "xlsx", "xls"]
)

try:
    coef_df, range_df, center, risk_cut = load_model(
        uploaded_coef, uploaded_range, uploaded_cut
    )
except Exception as exc:
    st.error(f"Unable to load model files: {exc}")
    st.stop()

st.title("Cardiometabolic Multimorbidity Risk Prediction")
st.write(
    "Enter patient information to calculate the Cox model score, risk category, "
    "and contributing risk indicators."
)

with st.expander("Model information", expanded=False):
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Model", "RCS-LASSO-Cox")
    info_col2.metric("Risk cut-off", f"{risk_cut:.4f}")
    info_col3.metric("Center", f"{center:.4f}")

raw_values = get_user_inputs(coef_df, range_df)
st.divider()

if st.button("Calculate Risk", type="primary", use_container_width=True):
    score, detail_df = calculate_score(raw_values, coef_df, range_df, center)
    risk_label = "High-risk" if score > risk_cut else "Low-risk"
    active_df = detail_df.loc[detail_df["Active"]].copy()
    active_df["Contribution"] = active_df["Contribution"].round(6)

    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Model Score", f"{score:.4f}")
    result_col2.metric("Risk Cut-off", f"{risk_cut:.4f}")
    result_col3.metric("Risk Classification", risk_label)

    if risk_label == "High-risk":
        st.error("High-risk: the model score is above the RCS-LASSO-Cox cut-off.")
    else:
        st.success("Low-risk: the model score is at or below the RCS-LASSO-Cox cut-off.")

    st.subheader("Contributing Risk Indicators")
    if active_df.empty:
        st.info("No active risk indicators were identified from the entered values.")
    else:
        display_df = active_df[
            ["Variable", "Observed value", "Indicator", "Coefficient", "Contribution"]
        ].sort_values("Contribution", ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        chart_df = display_df.set_index("Variable")["Contribution"]
        st.bar_chart(chart_df)

    all_results = detail_df.drop(columns=["Active", "Rule"])
    result_csv = all_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Patient Risk Report",
        data=result_csv,
        file_name="cardiometabolic_risk_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption(
    "This tool is intended for research and decision support. It does not replace "
    "clinical assessment or professional medical advice."
)