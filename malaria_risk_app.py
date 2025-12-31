import streamlit as st
import joblib
import pandas as pd

model_path = r"C:\Users\LENOVO\Desktop\sk\Malaria_Modell.pkl"
encoder_path = r"C:\Users\LENOVO\Desktop\sk\Malaria_Encoder.pkl"
model = joblib.load(model_path)
encoder = joblib.load(encoder_path)

st.set_page_config(page_title="Malaria Risk Screening App")

st.title("🦟 Malaria Risk Screening System")
st.markdown("""
⚠ **Disclaimer:**  
This tool is for **educational and screening purposes only**.  
It does **not** replace professional medical diagnosis.
""")

st.subheader("Enter Patient Information")
#st.title("🦟 Malaria Risk Predictor")
#st.image("mosquito.jpg", caption="Be aware of malaria-carrying mosquitoes", use_column_width=True)

st.warning("This tool predicts malaria risk based on symptoms only.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=25)
fever = st.selectbox("Fever", ["Yes", "No"])
headache = st.selectbox("Headache", ["Yes", "No"])
chills = st.selectbox("Chills", ["Yes", "No"])
sweats = st.selectbox("Sweats", ["Yes", "No"])
fatigue = st.selectbox("Fatigue", ["Yes", "No"])

# Predict button
if st.button("Predict Malaria Risk"):
    # Prepare input data
    new_data = pd.DataFrame({
        'Age': [age],
        'Fever': [fever],
        'Headache': [headache],
        'Chills': [chills],
        'Sweats': [sweats],
        'Fatigue': [fatigue]
    })

    # Encode categorical features
    new_encoded = pd.DataFrame(encoder.transform(new_data[['Fever', 'Headache', 'Chills', 'Sweats', 'Fatigue']]), 
                               columns=encoder.get_feature_names_out(['Fever', 'Headache', 'Chills', 'Sweats', 'Fatigue']))
    new_final = pd.concat([new_data[['Age']].reset_index(drop=True), new_encoded], axis=1)

    # Predict
    prediction = model.predict(new_final)[0]
    st.success(f"Predicted Malaria Risk: {prediction}")

    # Advisory
    st.info("⚠️ Please consult a healthcare professional for confirmation and proper diagnosis.")













