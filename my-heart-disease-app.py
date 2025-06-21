import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import io
import time

# Load and Clean Data (with flexible columns)
@st.cache_data
def load_and_clean_data(file_content):
    df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
    
    # Define expected columns
    all_columns = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours', 
                   'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level',
                   'Gender', 'Exercise Habits', 'Smoking', 'Family Heart Disease', 
                   'Diabetes', 'High Blood Pressure', 'Low HDL Cholesterol', 
                   'High LDL Cholesterol', 'Alcohol Consumption', 'Stress Level', 
                   'Sugar Consumption', 'Heart Disease Status']
    numerical_columns = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours', 
                        'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level']
    categorical_columns = ['Gender', 'Exercise Habits', 'Smoking', 'Family Heart Disease', 
                          'Diabetes', 'High Blood Pressure', 'Low HDL Cholesterol', 
                          'High LDL Cholesterol', 'Alcohol Consumption', 'Stress Level', 
                          'Sugar Consumption']
    
    # Validate minimum requirements (e.g., target and some features)
    required_min = ['Age', 'Gender', 'Heart Disease Status']  # Minimum for model to work
    missing_min = [col for col in required_min if col not in df.columns]
    if missing_min:
        st.error(f"Missing critical columns: {missing_min}. At least {required_min} are required.")
        st.stop()
    
    # Warn about missing columns
    present_columns = [col for col in all_columns if col in df.columns]
    missing_columns = [col for col in all_columns if col not in df.columns]
    if missing_columns:
        st.warning(f"Missing columns: {missing_columns}. Proceeding with available columns: {present_columns}")
    
    # Validate data types for present columns
    for column in [col for col in numerical_columns if col in df.columns]:
        if not pd.api.types.is_numeric_dtype(df[column]):
            st.error(f"Column '{column}' must be numeric. Please check your dataset.")
            st.stop()
    for column in [col for col in categorical_columns if col in df.columns]:
        if not pd.api.types.is_object_dtype(df[column]) and not pd.api.types.is_categorical_dtype(df[column]):
            st.error(f"Column '{column}' must be categorical. Please check your dataset.")
            st.stop()
    
    # Fill missing values for present columns
    for column in [col for col in numerical_columns if col in df.columns]:
        df[column] = df[column].fillna(df[column].mean())
    for column in [col for col in categorical_columns if col in df.columns]:
        df[column] = df[column].fillna(df[column].mode()[0])
        original_values[column] = df[column].dropna().unique().tolist()
    
    # Encode categorical columns that are present
    label_encoders = {}
    for column in [col for col in categorical_columns if col in df.columns]:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    # Encode target (required)
    le_target = LabelEncoder()
    df['Heart Disease Status'] = le_target.fit_transform(df['Heart Disease Status'])
    
    # Warn about high missing value percentage
    missing_percent = df.isnull().mean() * 100
    if any(missing_percent > 20):
        st.warning("Some columns had more than 20% missing values. Imputation applied.")
    
    return df, label_encoders, le_target, [col for col in categorical_columns if col in df.columns], original_values

# Feature Engineering
def engineer_features(df):
    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=['<30', '30-50', '>50'], include_lowest=True)
    if 'BMI' in df.columns:
        df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'], include_lowest=True)
    if 'Cholesterol Level' in df.columns:
        df['Cholesterol_Risk'] = pd.cut(df['Cholesterol Level'], bins=[0, 200, 239, 1000], labels=['Normal', 'Borderline', 'High'], include_lowest=True)
    for column in ['Age_Group', 'BMI_Category', 'Cholesterol_Risk']:
        if column in df.columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
    return df

# Exploratory Data Analysis
def exploratory_data_analysis(df):
    st.subheader("Exploratory Data Analysis")
    st.write("Dataset Overview:", df.describe())
    st.write("Missing Values:", df.isnull().sum())
    status_counts = df['Heart Disease Status'].value_counts()
    st.bar_chart(status_counts)

# Train and Evaluate Model
@st.cache_resource
def train_evaluate_model(X, y):
    start = time.time()
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X, y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.write(f"Model training time: {time.time() - start:.2f} seconds")
    return model, scaler

# Streamlit App
def main():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Ayhormie's Heart Diagnosis</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.image("https://static.streamlit.io/badges/streamlit_badge_black_white.png", width=150)
    
    st.sidebar.header("Instructions")
    st.sidebar.write("1. Upload a 'heart_disease.csv' file with required columns.")
    st.sidebar.write("2. Fill in the input fields.")
    st.sidebar.write("3. Click 'Predict' to see the result.")
    
    uploaded_file = st.file_uploader("Upload heart_disease.csv", type=["csv"])
    if uploaded_file is not None:
        file_content = uploaded_file.read()
        start = time.time()
        df, label_encoders, le_target, categorical_columns, original_values = load_and_clean_data(file_content)
        st.write(f"Data cleaning time: {time.time() - start:.2f} seconds")
        
        start = time.time()
        df = engineer_features(df)
        st.write(f"Feature engineering time: {time.time() - start:.2f} seconds")
        exploratory_data_analysis(df)
        
        X = df.drop('Heart Disease Status', axis=1)
        y = df['Heart Disease Status']
        
        start = time.time()
        model, scaler = train_evaluate_model(X, y)
        st.write(f"Total model training time: {time.time() - start:.2f} seconds (cached on first run)")
        
        st.subheader("Predict Heart Disease")
        with st.form("prediction_form"):
            input_data = {}
            numerical_columns = [col for col in ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours', 
                                               'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level'] if col in df.columns]
            categorical_columns = [col for col in ['Gender', 'Exercise Habits', 'Smoking', 'Family Heart Disease', 
                                                  'Diabetes', 'High Blood Pressure', 'Low HDL Cholesterol', 
                                                  'High LDL Cholesterol', 'Alcohol Consumption', 'Stress Level', 
                                                  'Sugar Consumption'] if col in df.columns]
            
            for column in numerical_columns:
                input_data[column] = st.number_input(f"{column}", value=float(df[column].mean()))
            
            for column in categorical_columns:
                input_data[column] = st.selectbox(f"{column}", original_values.get(column, ['N/A']))
            
            predict_button = st.form_submit_button("Predict")
        
        if predict_button:
            if not input_data:
                st.error("No input data provided. Please fill in all fields.")
                return
            
            input_df = pd.DataFrame([input_data])
            st.write("Input DataFrame:", input_df)
            
            # Encode input data using label encoders
            for column in input_df.columns:
                if column in label_encoders or column in ['Age_Group', 'BMI_Category', 'Cholesterol_Risk']:
                    le = label_encoders.get(column, LabelEncoder())
                    input_df[column] = le.transform(input_df[column])
            
            input_df = input_df.reindex(columns=X.columns, fill_value=0)
            st.write("Reindexed Input DataFrame:", input_df)
            
            try:
                start = time.time()
                input_scaled = scaler.transform(input_df)
                st.write(f"Scaling time: {time.time() - start:.2f} seconds")
                prediction = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled)
                st.write(f"Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
                st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")
            except ValueError as e:
                st.error(f"Prediction failed: {str(e)}")
                st.write("Input DataFrame shape:", input_df.shape)
                st.write("Training Data columns:", X.columns.tolist())
        
        st.subheader("Provide Feedback")
        with st.form("feedback_form"):
            feedback_text = st.text_area("Your feedback or suggestions:")
            submit_feedback = st.form_submit_button("Submit Feedback")
            if submit_feedback and feedback_text:
                st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()
