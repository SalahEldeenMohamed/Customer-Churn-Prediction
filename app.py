import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models and preprocessing tools
logistic_model = joblib.load(r'C:\\Users\\dell\\Downloads\\Customer_churn_prediction\\notebook\\logistic_regression_model.joblib')
rf_model = joblib.load(r'C:\\Users\\dell\\Downloads\\Customer_churn_prediction\\notebook\\random_forest_model.joblib')
svc_model = joblib.load(r'C:\\Users\\dell\\Downloads\\Customer_churn_prediction\\notebook\\SVC_model.joblib')
scaler = joblib.load(r'C:\\Users\\dell\\Downloads\\Customer_churn_prediction\\notebook\\Scaler.joblib')
ohe = joblib.load(r'C:\\Users\\dell\\Downloads\\Customer_churn_prediction\\notebook\\OneHotEncoder.joblib')

# Streamlit app
st.title("Customer Churn Prediction Web App")

st.write("Fill in the customer details below to predict the likelihood of churn.")

# Input fields
input_data = {}
input_data['gender'] = st.selectbox("Gender", ['Male', 'Female'])
input_data['SeniorCitizen'] = st.selectbox("Senior Citizen", [0, 1])
input_data['Partner'] = st.selectbox("Partner", ['Yes', 'No'])
input_data['Dependents'] = st.selectbox("Dependents", ['Yes', 'No'])
input_data['PhoneService'] = st.selectbox("Phone Service", ['Yes', 'No'])
input_data['MultipleLines'] = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
input_data['InternetService'] = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
input_data['OnlineSecurity'] = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
input_data['OnlineBackup'] = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
input_data['DeviceProtection'] = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
input_data['StreamingTV'] = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
input_data['StreamingMovies'] = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
input_data['Contract'] = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
input_data['PaperlessBilling'] = st.selectbox("Paperless Billing", ['Yes', 'No'])
input_data['PaymentMethod'] = st.selectbox("Payment Method", [
    'Credit card (automatic)', 'Bank transfer (automatic)', 'Mailed check', 'Electronic check'
])
input_data['TechSupport'] = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
input_data['MonthlyCharges'] = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
input_data['tenure'] = st.number_input("Tenure (in months)", min_value=0, step=1)

# Select the model to use
model_choice = st.selectbox("Select Model", ['Logistic Regression', 'Random Forest', 'SVC'])

# Predict button
if st.button("Predict Churn"):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess categorical features using one-hot encoding
    categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                            'PaymentMethod', 'TechSupport']
    encoded_features = ohe.transform(input_df[categorical_features]).toarray()

    # Scale numerical features
    numerical_features = ['MonthlyCharges', 'tenure']
    scaled_features = scaler.transform(input_df[numerical_features])

    # Combine processed features
    processed_input = np.hstack([encoded_features, scaled_features])

    # Select and use the chosen model
    if model_choice == 'Logistic Regression':
        model = logistic_model
    elif model_choice == 'Random Forest':
        model = rf_model
    elif model_choice == 'SVC':
        model = svc_model

    # Make prediction
    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0] if hasattr(model, 'predict_proba') else None

    # Display results
    st.write(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
    if probability is not None:
        st.write(f"Probability of Churn: {probability[1]:.2f}")
