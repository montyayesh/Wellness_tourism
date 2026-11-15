import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_model_v1.joblib")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Wellness Tourism Package Prediction")
st.write("Fill the customer details below to predict if they'll purchase the Wellness Tourism Package.")

# Collect user input
col1, col2 = st.columns(2)
with col1:
    Age = st.slider("Age", 18, 70, 30)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, value=30000.0, step=500.0)
    Passport = st.selectbox("Has Passport?", ["Yes", "No"])
with col2:
    OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
    NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 2)
    NumberOfChildrenVisiting = st.slider("Number of Children Visiting (below age 5)", 0, 3, 0)
    NumberOfTrips = st.slider("Number of Trips per Year", 1, 22, 3)
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    DurationOfPitch = st.slider("Duration of Pitch (mins)", 5, 130, 15)
    NumberOfFollowups = st.slider("Number of Follow-ups", 1, 6, 3)
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

# ----------------------------
# Prepare input data
# ----------------------------
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# Set the classification threshold (same value used in train.py)
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prob = model.predict_proba(input_data)[0, 1]
    pred = int(prob >= classification_threshold)
    if pred == 1:
        st.success(f"Prediction: Customer will purchase the package (probability {prob:.2%})")
    else:
        st.warning(f"Prediction: Customer is unlikely to purchase (probability {prob:.2%})")
