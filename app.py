import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
import joblib

# ML PIPELINE SETUP (ADJUSTED FOR LINEAR REGRESSION) 
@st.cache_resource
def setup_ml_pipeline(data_path):
    # --- Data Preparation (Same as Part 1) ---
    df = pd.read_csv(data_path)
    df.drop_duplicates(inplace=True) 
    
    # Feature Engineering: Log Transformation for Targets
    df['log_actual_cost'] = np.log1p(df['Actual_Cost'])
    df['log_actual_duration'] = np.log1p(df['Actual_Duration'])

    # Define columns to drop 
    columns_to_drop = [
        'Project_ID', 'Start_Date', 'End_Date', 
        'Actual_Cost', 'Actual_Duration', 'Cost_Overrun', 
        'Schedule_Deviation', 'Planned_Cost', 'Planned_Duration'
    ]
    df_features = df.drop(columns=columns_to_drop, errors='ignore')

    # One-Hot Encoding
    categorical_cols = ['Project_Type', 'Location', 'Weather_Condition', 'Risk_Level']
    df_encoded = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)

    # Define Features (X) and Targets (Y)
    X = df_encoded.drop(columns=['log_actual_cost', 'log_actual_duration'])
    Y_cost = df_encoded['log_actual_cost']
    Y_duration = df_encoded['log_actual_duration']
    
    # Split Data
    X_train, _, Y_train_cost, _ = train_test_split(X, Y_cost, test_size=0.2, random_state=42)
    _, _, Y_train_duration, _ = train_test_split(X, Y_duration, test_size=0.2, random_state=42)
    
    # --- Scaling (Fit Scaler on Training Data) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) 
    
    # Training Final linear regression Models 
    
    # Cost Model
    lr_cost = LinearRegression()
    lr_cost.fit(X_train_scaled, Y_train_cost)
    
    # Duration Model
    lr_duration = LinearRegression()
    lr_duration.fit(X_train_scaled, Y_train_duration)
    
    # Return the new models
    return lr_cost, lr_duration, scaler, X_train.columns.tolist()

# Execute the pipeline and load assets
try:
    COST_MODEL, DURATION_MODEL, SCALER, FEATURE_NAMES = setup_ml_pipeline("bim_ai_civil_engineering_dataset.csv")
except FileNotFoundError:
    st.error("Error: The file 'bim_ai_civil_engineering_dataset.csv' was not found.")
    st.stop()
except Exception as e:
    st.error(f"Error during ML pipeline setup: {e}")
    st.stop()


# INPUT PREPROCESSING ENGINE (UPDATED FOR SCALING) 
def prepare_input_for_model(raw_input_data, feature_names, scaler):
    """Transforms raw user input into the correctly encoded and SCALED vector."""

    # Define original categorical columns
    categorical_cols = ['Project_Type', 'Location', 'Weather_Condition', 'Risk_Level']
    
    # 1. Apply One-Hot Encoding (using drop_first=True to match training)
    input_encoded = pd.get_dummies(raw_input_data, columns=categorical_cols, drop_first=True)

    # 2. Align Columns (CRITICAL STEP)
    # Create an empty DataFrame matching the training data structure
    final_input = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Update the empty frame with the values from the user's encoded input
    for col in input_encoded.columns:
        if col in final_input.columns:
            final_input[col] = input_encoded[col].values
            
    # 3. Scaling (CRITICAL FOR LINEAR REGRESSION)
    # Transform the final, aligned input using the fitted scaler
    scaled_array = scaler.transform(final_input)
    
    # Return the scaled array for prediction
    return scaled_array

# 3. STREAMLIT UI AND INPUT COLLECTION 

st.title("🏗️ ML Project Cost & Duration Predictor")
st.markdown("Forecast construction project metrics using our predictive **Linear Regression** models.")

st.sidebar.header("Project Parameters (What-If Scenario)")

# Collect user inputs for all features

# A. Categorical Features (for selection)
PROJECT_TYPES = ['Tunnel', 'Dam', 'Building', 'Road', 'Bridge']
LOCATIONS = ['Houston', 'Seattle', 'Los Angeles', 'New York', 'Chicago']
WEATHER = ['Sunny', 'Cloudy', 'Rainy', 'Stormy', 'Snowy']
RISK = ['High', 'Medium', 'Low']

project_type = st.sidebar.selectbox("Project Type:", PROJECT_TYPES)
location = st.sidebar.selectbox("Location:", LOCATIONS)
weather = st.sidebar.selectbox("Weather Condition:", WEATHER)
risk_level = st.sidebar.selectbox("Risk Level:", RISK)

# B. Numerical Features (using sliders/number inputs)
st.sidebar.markdown("---")
st.sidebar.subheader("Sensor and Operational Data")

vibration = st.sidebar.number_input("Vibration Level:", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
crack_width = st.sidebar.number_input("Crack Width:", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
load_capacity = st.sidebar.number_input("Load Bearing Capacity:", min_value=0.0, max_value=500.0, value=250.0, step=1.0)
temperature = st.sidebar.number_input("Temperature:", min_value=-10.0, max_value=50.0, value=20.0, step=0.5)
humidity = st.sidebar.number_input("Humidity:", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
air_quality = st.sidebar.slider("Air Quality Index:", 50, 300, 150)
energy_consumption = st.sidebar.number_input("Energy Consumption:", min_value=5000.0, max_value=50000.0, value=25000.0)
material_usage = st.sidebar.number_input("Material Usage:", min_value=100.0, max_value=1000.0, value=500.0)
labor_hours = st.sidebar.number_input("Labor Hours:", min_value=1000, max_value=10000, value=5000)
equipment_util = st.sidebar.slider("Equipment Utilization (%):", 40.0, 100.0, 75.0)
accident_count = st.sidebar.slider("Accident Count:", 0, 10, 5)
safety_risk = st.sidebar.number_input("Safety Risk Score:", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
image_analysis = st.sidebar.slider("Image Analysis Score (%):", 50.0, 100.0, 75.0)
anomaly_detected = st.sidebar.selectbox("Anomaly Detected (1=Yes, 0=No):", [0, 1])
completion_percent = st.sidebar.slider("Completion Percentage:", 10.0, 100.0, 50.0)


# PREDICTION LOGIC 
if st.sidebar.button("Generate Forecast"):
    # Collect all inputs into a single DataFrame row for preprocessing
    raw_input_data = pd.DataFrame({
        'Project_Type': [project_type],
        'Location': [location],
        'Weather_Condition': [weather],
        'Risk_Level': [risk_level],
        'Vibration_Level': [vibration],
        'Crack_Width': [crack_width],
        'Load_Bearing_Capacity': [load_capacity],
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Air_Quality_Index': [air_quality],
        'Energy_Consumption': [energy_consumption],
        'Material_Usage': [material_usage],
        'Labor_Hours': [labor_hours],
        'Equipment_Utilization': [equipment_util],
        'Accident_Count': [accident_count],
        'Safety_Risk_Score': [safety_risk],
        'Image_Analysis_Score': [image_analysis],
        'Anomaly_Detected': [anomaly_detected],
        'Completion_Percentage': [completion_percent]
    })
    
    try:
        # Preprocess the input data using the function and cached assets
        # This returns the scaled numpy array (essential for Linear Regression)
        processed_input_array = prepare_input_for_model(raw_input_data, FEATURE_NAMES, SCALER)
        
        # Prediction on the scaled array
        cost_log_pred = COST_MODEL.predict(processed_input_array)[0]
        duration_log_pred = DURATION_MODEL.predict(processed_input_array)[0]
        
        # Inverse Transformation
        cost_pred = np.expm1(cost_log_pred)
        duration_pred = np.expm1(duration_log_pred)
        
        # Displaying Results 
        st.header("Automated Forecasts")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Project Cost (CAD)", f"${cost_pred:,.2f}")
        with col2:
            st.metric("Predicted Project Duration (Days)", f"{int(duration_pred):,} days")
            
        st.success("Forecast generated successfully for the 'What-If' scenario!")
        
    except Exception as e:

        st.error(f"Prediction error: {e}")
