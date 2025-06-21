import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import io

@st.cache_data
def load_and_clean_data(file_content):
    df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
    
    # Explicitly define columns
    numerical_columns = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours', 
                        'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level']
    categorical_columns = ['Gender', 'Exercise Habits', 'Smoking', 'Family Heart Disease', 
                          'Diabetes', 'High Blood Pressure', 'Low HDL Cholesterol', 
                          'High LDL Cholesterol', 'Alcohol Consumption', 'Stress Level', 
                          'Sugar Consumption']
    
    # Fill missing values
    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].mean())
    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])
    
    # Encode categorical columns (delay if needed for feature engineering)
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    # Encode target
    le_target = LabelEncoder()
    df['Heart Disease Status'] = le_target.fit_transform(df['Heart Disease Status'])
    
    # Optional: Warn about high missing value percentage
    missing_percent = df.isnull().mean() * 100
    if any(missing_percent > 20):
        st.warning("Some columns had more than 20% missing values. Imputation applied.")
    
    return df, label_encoders, le_target, categorical_columns

# Feature Engineering
def engineer_features(df):
    df['BMI_Category'] = pd.qcut(df['BMI'], q=3, labels=['Low', 'Medium', 'High']).astype('category')
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 80], labels=['Young', 'Middle', 'Senior']).astype('category')
    df['Cholesterol_Risk'] = np.where(df['Cholesterol Level'] > 240, 1, 0)
    
    for column in ['BMI_Category', 'Age_Group']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
    
    return df

# EDA and Visualization
def exploratory_data_analysis(df):
    st.subheader("Exploratory Data Analysis")
    st.write("### Summary Statistics")
    st.write(df.describe())
    
    st.write("### Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot()
    
    st.write("### Feature Distributions")
    for column in df.columns[:-1]:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=column, hue='Heart Disease Status', multiple="stack")
        st.pyplot()

# Model Training and Evaluation with Tuning
def train_evaluate_model(X, y):
    st.subheader("Model Training and Evaluation")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    st.write(f"Best Parameters: {grid_search.best_params_}")
    st.write(f"Best Cross-Validation Score: {grid_search.best_score_:.2f}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Test Accuracy: {accuracy:.2f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    # Cross-validation score
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)
    st.write(f"Cross-Validation Scores: {cv_scores}")
    st.write(f"Mean CV Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    
    return best_model, scaler

# Streamlit App
def main():
    # Styling: Centered title with color
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Ayhormie's Heart Diagnosis</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar instructions
    st.sidebar.header("Instructions")
    st.sidebar.write("1. Upload a 'heart_disease.csv' file with required columns.")
    st.sidebar.write("2. Fill in the input fields.")
    st.sidebar.write("3. Click 'Predict' to see the result.")
    
    uploaded_file = st.file_uploader("Upload heart_disease.csv", type="csv")
    if uploaded_file is not None:
        file_content = uploaded_file.read()
        df, label_encoders, le_target, categorical_columns = load_and_clean_data(file_content)
        
        df = engineer_features(df)
        exploratory_data_analysis(df)
        
        X = df.drop('Heart Disease Status', axis=1)
        y = df['Heart Disease Status']
        
        model, scaler = train_evaluate_model(X, y)
        
        st.subheader("Predict Heart Disease")
        with st.form("prediction_form"):
            input_data = {}
            numerical_columns = ['Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours', 
                                'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level']
            categorical_columns = ['Gender', 'Exercise Habits', 'Smoking', 'Family Heart Disease', 
                                  'Diabetes', 'High Blood Pressure', 'Low HDL Cholesterol', 
                                  'High LDL Cholesterol', 'Alcohol Consumption', 'Stress Level', 
                                  'Sugar Consumption']
            
            for column in numerical_columns:
                input_data[column] = st.number_input(f"{column}", value=float(df[column].mean()))
            
            for column in categorical_columns:
                unique_values = df[column].dropna().unique()
                input_data[column] = st.selectbox(f"{column}", unique_values)
            
            predict_button = st.form_submit_button("Predict")
        
        if predict_button:
            if not input_data:
                st.error("No input data provided. Please fill in all fields.")
                return
            
            input_df = pd.DataFrame([input_data])
            st.write("Input DataFrame:", input_df)
            
            for column in input_df.columns:
                if column in label_encoders or column in ['BMI_Category', 'Age_Group']:
                    if column in categorical_columns:
                        le = label_encoders[column]
                        input_df[column] = le.transform(input_df[column])
                    else:
                        le = LabelEncoder()
                        input_df[column] = le.fit_transform(input_df[column])
            
            input_df = input_df.reindex(columns=X.columns, fill_value=0)
            st.write("Reindexed Input DataFrame:", input_df)
            
            try:
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled)
                st.write(f"Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
                st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")
            except ValueError as e:
                st.error(f"Prediction failed: {str(e)}")
                st.write("Input DataFrame shape:", input_df.shape)
                st.write("Training Data columns:", X.columns.tolist())
        
        # Feedback section
        st.subheader("Provide Feedback")
        with st.form("feedback_form"):
            feedback_text = st.text_area("Your feedback or suggestions:")
            submit_feedback = st.form_submit_button("Submit Feedback")
            if submit_feedback and feedback_text:
                st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()
