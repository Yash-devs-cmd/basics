import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Insurance Charges Predictor", layout="wide")

# -------------------------
# Load saved scaler & model
# -------------------------
scaler = joblib.load("scaler.pkl")
best_model = joblib.load("best_model.pkl")

# -------------------------
# Page Title
# -------------------------
st.title("💰 Insurance Charges Predictor")
st.write("Fill the form below to predict insurance cost.")

# -------------------------
# FORM UI
# -------------------------
with st.form("insurance_form"):
    st.markdown("### Patient Information Form")

    col1, col2 = st.columns(2)

    # Left side inputs
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0, step=0.1)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

    # Right side inputs
    with col2:
        is_female = st.checkbox("Female")
        is_smoker = st.checkbox("Smoker")

        region = st.selectbox(
            "Region",
            ["northwest", "southeast", "southwest", "northeast"]
        )

        bmi_category = st.selectbox(
            "BMI Category",
            ["Normal", "Overweight", "Obese", "Underweight"]
        )

    # Form submit button
    submitted = st.form_submit_button("Predict Insurance Charges")

# -------------------------
# Prediction Logic
# -------------------------
if submitted:
    # Encoding
    region_nw = 1 if region == "northwest" else 0
    region_se = 1 if region == "southeast" else 0
    region_sw = 1 if region == "southwest" else 0

    bmi_overweight = 1 if bmi_category == "Overweight" else 0
    bmi_obese = 1 if bmi_category == "Obese" else 0
    bmi_underweight = 1 if bmi_category == "Underweight" else 0

    x_input = np.array([[
        age, bmi, children,
        int(is_female), int(is_smoker),
        region_nw, region_se, region_sw,
        bmi_overweight, bmi_obese, bmi_underweight
    ]])

    # Scale numeric columns
    x_input[:, :3] = scaler.transform(x_input[:, :3])

    # Predict
    prediction = best_model.predict(x_input)[0]

    st.subheader("🔮 Predicted Insurance Charges")
    st.success(f"**₹{prediction:,.2f}**")
