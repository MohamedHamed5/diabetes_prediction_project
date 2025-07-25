

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import uuid
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Page setup
st.set_page_config(layout="wide", page_title="Diabetes Prediction", page_icon="\U0001F3E5")

# CSS Styling
st.markdown("""
    <style>
        body { background-color: #f0f4f8; }
        .main { background-color: #ffffff; padding: 30px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #004080; font-weight: 700; }
        .sidebar .css-1d391kg { background-color: #e6f0ff !important; border-radius: 10px; }
        .stButton>button {
            background-color: #004080; color: white;
            border-radius: 8px; padding: 12px 25px;
            font-weight: bold; font-size: 16px;
        }
        .footer-box {
            background-color: #f9f9f9; padding: 20px;
            border-radius: 12px; font-size: 16px;
            color: #333; margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar - Input form
with st.sidebar:
    st.title("\U0001F4DC Enter Your Health Information")
    st.markdown("### Fill in your details:")

    patient_name = st.text_input("Patient Name")

    columns = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
        'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]

    inputs = {}
    for col in columns:
        if col in ['BMI', 'MentHlth', 'PhysHlth']:
            inputs[col] = st.number_input(f"{col}:", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        elif col == 'GenHlth':
            genhlth_map = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
            inputs[col] = st.selectbox("General Health:", options=list(genhlth_map.keys()), format_func=lambda x: genhlth_map[x])
        elif col == 'Age':
            age_map = {1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44", 6: "45–49", 7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69", 11: "70–74", 12: "75–79", 13: "80 or older"}
            inputs[col] = st.selectbox("Age Group:", options=list(age_map.keys()), format_func=lambda x: age_map[x])
        elif col == 'Education':
            education_map = {1: "Never attended school", 2: "Elementary school (1–8)", 3: "Some high school (9–11)", 4: "High school graduate (12th grade)", 5: "Some college, no degree", 6: "2-year college degree (Associate's)", 7: "4-year college degree (Bachelor's)", 8: "Graduate-level education (Master's or higher)"}
            inputs[col] = st.selectbox("Education Level:", options=list(education_map.keys()), format_func=lambda x: education_map[x])
        elif col == 'Income':
            income_map = {1: "Less than $10,000", 2: "$10,000 – $15,000", 3: "$15,000 – $20,000", 4: "$20,000 – $25,000", 5: "$25,000 – $35,000", 6: "$35,000 – $50,000", 7: "$50,000 – $75,000", 8: "More than $75,000"}
            inputs[col] = st.selectbox("Income Level:", options=list(income_map.keys()), format_func=lambda x: income_map[x])
        else:
            inputs[col] = st.selectbox(f"{col}:", options=[0, 1])

# Main Content
st.markdown("<div class='main'>", unsafe_allow_html=True)

st.markdown("<h1> \U0001F3E5 Diabetes Prediction Model</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:18px; color:#1E90FF;'>↪ <b>Diabetes doesn't wait. Neither should you.</b></p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #004080;'>", unsafe_allow_html=True)

st.markdown("""
<div style="background-color:#A3C9FF; padding: 15px; border-radius:10px; font-size:16px;">
     <b>\U0001F4CC Enter your health information on the left and click 'Predict Now 🎯' to see your results.</b>
</div>
""", unsafe_allow_html=True)

if st.button("Predict Now 🎯"):
    input_values = np.array([list(inputs.values())])
    input_scaled = scaler.transform(input_values)
    prediction = int(model.predict(input_scaled)[0])
    probabilities = model.predict_proba(input_scaled)[0] * 100

    labels = ['Not Diabetic', 'Prediabetic', 'Diabetic']
    colors = ['🟢', '🟡', '🔴']
    risk = labels[prediction]

    st.markdown("<h2>\U0001F4CA Prediction Results</h2>", unsafe_allow_html=True)

    if prediction == 0:
        st.success(f"✅ This person is {risk}.")
    elif prediction == 1:
        st.warning(f"⚠ This person is {risk}.")
    else:
        st.error(f"\U0001F6A9 This person is {risk}.")

    col1, col2, col3 = st.columns(3)
    col1.metric(label=f"🟢 Not Diabetic", value=f"{probabilities[0]:.2f}%")
    col2.metric(label=f"🟡 Prediabetic", value=f"{probabilities[1]:.2f}%")
    col3.metric(label=f"🔴 Diabetic", value=f"{probabilities[2]:.2f}%")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=probabilities, y=labels, palette="coolwarm", ax=ax)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    ax.set_title("Prediction Probability Distribution")
    st.pyplot(fig)

    st.markdown(f"""
    ### \U0001F4CC Interpretation:
    The model estimates there's a {probabilities[prediction]:.2f}% chance that this person is {risk}.
    Disclaimer: This is a machine learning prediction, not a clinical diagnosis.
    """)

    patient_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {**inputs, 'PatientName': patient_name, 'PatientID': patient_id, 'Prediction': risk, 'PredictionCode': prediction,
              'Probability_NotDiabetic': round(probabilities[0], 2),
              'Probability_Prediabetic': round(probabilities[1], 2),
              'Probability_Diabetic': round(probabilities[2], 2),
              'Timestamp': timestamp}

    df_new = pd.DataFrame([record])
    file_path = "patients_records.csv"
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(file_path, index=False)

    st.info(f"\U0001F4C2 Patient data saved with ID: {patient_id}")

    # Create PDF report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="Diabetes Prediction Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, txt=f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 10, txt=f"Patient ID: {patient_id}", ln=True)
    pdf.cell(0, 10, txt=f"Date: {timestamp}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Input Features:", ln=True)
    pdf.set_font("Arial", size=11)

    for key, val in inputs.items():
        pdf.cell(90, 8, f"{key}", border=1)
        pdf.cell(90, 8, str(val), border=1, ln=True)

    pdf.ln(3)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Prediction Results:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(90, 8, "Not Diabetic", border=1)
    pdf.cell(90, 8, f"{probabilities[0]:.2f}%", border=1, ln=True)
    pdf.cell(90, 8, "Prediabetic", border=1)
    pdf.cell(90, 8, f"{probabilities[1]:.2f}%", border=1, ln=True)
    pdf.cell(90, 8, "Diabetic", border=1)
    pdf.cell(90, 8, f"{probabilities[2]:.2f}%", border=1, ln=True)
    pdf.cell(0, 8, f"Overall Prediction: {risk} ({probabilities[prediction]:.2f}%)", ln=True)

    # Create folder if not exist and save PDF
    report_folder = "Patient files"
    os.makedirs(report_folder, exist_ok=True)
    pdf_filename = f"diabetes_report_{patient_id}.pdf"
    pdf_path = os.path.join(report_folder, pdf_filename)
    pdf.output(pdf_path)

    with open(pdf_path, "rb") as f:
        st.download_button("\U0001F4C4 Download PDF Report", f, file_name=pdf_filename, mime="application/pdf")

# Footer
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("""
<div class='footer-box'>
    <b>Model Developed by <span style='color:#0072C6;'>AL-Arif Team</span> 🤖🤖</b><br>
    <b>As Part of the <span style='color:#388E3C;'>Machine Learning Track</span></b><br>
    <b>Under the Training Program of <span style='color:#D32F2F;'>NTI (National Telecommunication Institute)</span></b><br><br>
</div>
""", unsafe_allow_html=True)

