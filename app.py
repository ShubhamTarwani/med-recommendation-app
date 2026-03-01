import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Medical Diagnostic SaaS", layout="wide", page_icon="🩺")

# --- DATA LOADING (Cached for performance) ---
@st.cache_data
def load_and_standardize_data():
    # Load all files
    train_data = pd.read_csv('Main_Training_2026.csv')
    description = pd.read_csv('symptom_Description.csv')
    precaution = pd.read_csv('symptom_precaution.csv')
    meds = pd.read_csv('medications.csv')
    diets = pd.read_csv('diets.csv')
    workouts = pd.read_csv('workout_df.csv')

    # Normalize columns to 'Disease'
    for df in [train_data, description, precaution, meds, diets, workouts]:
        if 'prognosis' in df.columns:
            df.rename(columns={'prognosis': 'Disease'}, inplace=True)
        elif 'disease' in df.columns:
            df.rename(columns={'disease': 'Disease'}, inplace=True)
    
    return train_data, description, precaution, meds, diets, workouts

# Initialize Data
train_data, description, precaution, meds, diets, workouts = load_and_standardize_data()

# --- MODEL TRAINING & EVALUATION ---
@st.cache_resource
def train_model(data):
    X = data.select_dtypes(include=[np.number])
    y = data['Disease']
    
    # 80/20 Split to calculate real accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ACCURACY UPGRADE: Added class_weight='balanced' to handle skewed medical data
    # SPEED UPGRADE: Added n_jobs=-1 to utilize multi-core processing for the large dataset
    model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Calculate Accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, X.columns.values, accuracy

model, feature_names, model_accuracy = train_model(train_data)

# --- UI DESIGN ---
st.title("🩺 AI-Enabled SaaS Platform: Diagnostic Module")
st.caption("Developed for AIML Coursework - Nagpur 2026")

# Display the Accuracy Metric dynamically
st.markdown(f"**System Status:** Model Trained on 246k+ record")
st.divider()

# Sidebar for Input
st.sidebar.header("User Symptoms")
selected_symptoms = st.sidebar.multiselect(
    "Select your symptoms:",
    options=[s.replace("_", " ").capitalize() for s in feature_names]
)

if st.sidebar.button("Generate Report"):
    if not selected_symptoms:
        st.sidebar.warning("Please select at least one symptom.")
    else:
        # Vectorize Input
        input_vector = np.zeros(len(feature_names))
        for s in selected_symptoms:
            s_formatted = s.lower().replace(" ", "_")
            if s_formatted in feature_names:
                idx = np.where(feature_names == s_formatted)[0][0]
                input_vector[idx] = 1
        
        # Predict & Calculate Confidence
        prediction = model.predict([input_vector])[0]
        probabilities = model.predict_proba([input_vector])
        confidence = np.max(probabilities) * 100

        # Display Results header and Confidence Bar
        st.subheader(f"Results for: {prediction}")
        st.progress(int(confidence), text=f"AI Confidence Score: {confidence:.2f}%")
        st.write("") # Spacer
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Description**")
            try:
                desc_text = description[description['Disease'] == prediction]['Description'].values[0]
                st.write(desc_text)
            except IndexError:
                st.write("Description not available.")
            
            st.success("**Immediate Precautions**")
            try:
                prec_steps = precaution[precaution['Disease'] == prediction].iloc[0, 1:].dropna().tolist()
                for step in prec_steps:
                    st.write(f"- {step.strip().capitalize()}")
            except IndexError:
                st.write("- Precaution data not available.")

        with col2:
            st.warning("**Suggested Plan**")
            try:
                st.write(f"💊 **Medication:** {meds[meds['Disease'] == prediction]['Medication'].values[0]}")
                st.write(f"🥗 **Diet:** {diets[diets['Disease'] == prediction]['Diet'].values[0]}")
                st.write(f"🏋️ **Workout:** {workouts[workouts['Disease'] == prediction]['workout'].values[0]}")
            except IndexError:
                st.write("Treatment plan data is incomplete for this specific condition.")

st.markdown("---")
st.caption("Disclaimer: This is a student project and does not provide professional medical advice.")