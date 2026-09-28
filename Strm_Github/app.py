import streamlit as st
import pandas as pd
import joblib
import base64

# Page config
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

# Background image
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
        url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("images/titanic.jpg")

# Load model
model = joblib.load("models/titanic_best_model.pkl")

# Title
st.markdown("<h1 style='text-align:center;color:white;'>Titanic Survival Predictor</h1>", unsafe_allow_html=True)

# Input form
st.subheader("Passenger Information")
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
    sibsp = st.number_input("Siblings/Spouses", min_value=0, max_value=10, value=0)

with col2:
    parch = st.number_input("Parents/Children", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=30.0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

if st.button("Predict Survival", use_container_width=True):
    input_data = pd.DataFrame({
        "Pclass": [pclass], "Sex": [sex], "Age": [age],
        "SibSp": [sibsp], "Parch": [parch],
        "Fare": [fare], "Embarked": [embarked]
    })
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    survival_probability = probability[1]
    death_probability = probability[0]

    if prediction == 1:
        st.success("Passenger is predicted to SURVIVE.")
    else:
        st.error("Passenger is predicted NOT TO SURVIVE.")

    st.write(f"Survival Probability: {survival_probability:.2%}")
    st.write(f"Not Survival Probability: {death_probability:.2%}")
    st.progress(float(survival_probability))
