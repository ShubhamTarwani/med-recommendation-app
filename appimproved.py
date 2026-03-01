import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from pathlib import Path
import logging

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Medical Diagnostic SaaS", layout="wide", page_icon="🩺")

# --- DATA LOADING (Cached for performance) ---
@st.cache_data
def load_and_standardize_data():
    """
    Load and standardize all CSV files with proper error handling.
    Returns tuple of dataframes or None if loading fails.
    """
    base_path = Path(__file__).parent
    
    # Define required files
    files = {
        'train_data': 'Main_Training_2026.csv',
        'description': 'symptom_Description.csv',
        'precaution': 'symptom_precaution.csv',
        'meds': 'medications.csv',
        'diets': 'diets.csv',
        'workouts': 'workout_df.csv'
    }
    
    loaded_data = {}
    
    # Load files with error handling
    for key, filename in files.items():
        filepath = base_path / filename
        try:
            if not filepath.exists():
                st.error(f"❌ Error: File '{filename}' not found. Please ensure all data files are in the same directory.")
                logger.error(f"File not found: {filepath}")
                return None, None, None, None, None, None
            
            loaded_data[key] = pd.read_csv(filepath)
            logger.info(f"Successfully loaded {filename}")
            
            # Validate that file is not empty
            if loaded_data[key].empty:
                st.error(f"❌ Error: File '{filename}' is empty.")
                logger.error(f"Empty file: {filename}")
                return None, None, None, None, None, None
                
        except Exception as e:
            st.error(f"❌ Error loading '{filename}': {str(e)}")
            logger.error(f"Error loading {filename}: {e}")
            return None, None, None, None, None, None
    
    train_data = loaded_data['train_data']
    description = loaded_data['description']
    precaution = loaded_data['precaution']
    meds = loaded_data['meds']
    diets = loaded_data['diets']
    workouts = loaded_data['workouts']

    # Normalize columns to 'Disease'
    for df_name, df in [('train_data', train_data), ('description', description), 
                        ('precaution', precaution), ('meds', meds), 
                        ('diets', diets), ('workouts', workouts)]:
        if 'prognosis' in df.columns:
            df.rename(columns={'prognosis': 'Disease'}, inplace=True)
        elif 'disease' in df.columns:
            df.rename(columns={'disease': 'Disease'}, inplace=True)
    
    # Validate that 'Disease' column exists in train_data
    if 'Disease' not in train_data.columns:
        st.error("❌ Error: 'Disease' column not found in training data after standardization.")
        logger.error("Disease column missing in train_data")
        return None, None, None, None, None, None
    
    return train_data, description, precaution, meds, diets, workouts

# Initialize Data
train_data, description, precaution, meds, diets, workouts = load_and_standardize_data()

# Check if data loading was successful
if train_data is None:
    st.stop()

# --- MODEL TRAINING & EVALUATION ---
@st.cache_resource
def train_model(data):
    """
    Train the Random Forest model and calculate accuracy.
    Returns model, feature names, and accuracy score.
    """
    try:
        # Select numeric columns (symptoms)
        X = data.select_dtypes(include=[np.number])
        
        # Validate that we have features
        if X.empty:
            st.error("❌ Error: No numeric features found in training data.")
            logger.error("No numeric features in training data")
            return None, None, None
        
        # Validate that Disease column exists
        if 'Disease' not in data.columns:
            st.error("❌ Error: 'Disease' column not found in training data.")
            logger.error("Disease column missing")
            return None, None, None
        
        y = data['Disease']
        
        # Validate that we have data
        if len(X) == 0 or len(y) == 0:
            st.error("❌ Error: Training data is empty.")
            logger.error("Empty training data")
            return None, None, None
        
        # 80/20 Split to calculate real accuracy
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # ACCURACY UPGRADE: Added class_weight='balanced' to handle skewed medical data
        # SPEED UPGRADE: Added n_jobs=-1 to utilize multi-core processing for the large dataset
        model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1)
        
        with st.spinner("Training model... This may take a moment."):
            model.fit(X_train, y_train)
        
        # Calculate Accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model trained successfully. Accuracy: {accuracy:.4f}")
        
        return model, X.columns.values, accuracy
    
    except Exception as e:
        st.error(f"❌ Error during model training: {str(e)}")
        logger.error(f"Model training error: {e}")
        return None, None, None

model, feature_names, model_accuracy = train_model(train_data)

# Check if model training was successful
if model is None:
    st.stop()

# --- UI DESIGN ---
st.title("🩺 AI-Enabled SaaS Platform: Diagnostic Module")
st.caption("Developed for AIML Coursework - Nagpur 2026")

# Display the Accuracy Metric dynamically
record_count = len(train_data)
st.markdown(f"**System Status:** Model Trained on {record_count:,} records")
st.divider()

# Sidebar for Input
st.sidebar.header("User Symptoms")
selected_symptoms = st.sidebar.multiselect(
    "Select your symptoms:",
    options=[s.replace("_", " ").title() for s in feature_names]  # Changed to .title() for better formatting
)

if st.sidebar.button("Generate Report"):
    if not selected_symptoms:
        st.sidebar.warning("Please select at least one symptom.")
    else:
        # Vectorize Input
        input_vector = np.zeros(len(feature_names))
        matched_symptoms = []
        unmatched_symptoms = []
        
        for s in selected_symptoms:
            s_formatted = s.lower().replace(" ", "_")
            if s_formatted in feature_names:
                idx = np.where(feature_names == s_formatted)[0][0]
                input_vector[idx] = 1
                matched_symptoms.append(s)
            else:
                unmatched_symptoms.append(s)
        
        # Validate that at least one symptom was matched
        if np.sum(input_vector) == 0:
            st.error("❌ Error: None of the selected symptoms could be matched. Please try again.")
            logger.warning(f"No symptoms matched from: {selected_symptoms}")
        else:
            # Show feedback about matched/unmatched symptoms
            if unmatched_symptoms:
                st.sidebar.warning(f"⚠️ Note: {len(unmatched_symptoms)} symptom(s) could not be matched: {', '.join(unmatched_symptoms)}")
            
            # Predict & Calculate Confidence
            try:
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
                        desc_match = description[description['Disease'] == prediction]
                        if not desc_match.empty and 'Description' in desc_match.columns:
                            desc_text = desc_match['Description'].values[0]
                            st.write(desc_text)
                        else:
                            st.write("Description not available.")
                    except (IndexError, KeyError) as e:
                        st.write("Description not available.")
                        logger.warning(f"Description not found for {prediction}: {e}")
                    
                    st.success("**Immediate Precautions**")
                    try:
                        prec_match = precaution[precaution['Disease'] == prediction]
                        if not prec_match.empty:
                            prec_steps = prec_match.iloc[0, 1:].dropna().tolist()
                            if prec_steps:
                                for step in prec_steps:
                                    st.write(f"- {step.strip().capitalize()}")
                            else:
                                st.write("- Precaution data not available.")
                        else:
                            st.write("- Precaution data not available.")
                    except (IndexError, KeyError) as e:
                        st.write("- Precaution data not available.")
                        logger.warning(f"Precautions not found for {prediction}: {e}")

                with col2:
                    st.warning("**Suggested Plan**")
                    try:
                        # Medication
                        med_match = meds[meds['Disease'] == prediction]
                        if not med_match.empty and 'Medication' in med_match.columns:
                            st.write(f"💊 **Medication:** {med_match['Medication'].values[0]}")
                        else:
                            st.write("💊 **Medication:** Not available")
                        
                        # Diet
                        diet_match = diets[diets['Disease'] == prediction]
                        if not diet_match.empty and 'Diet' in diet_match.columns:
                            st.write(f"🥗 **Diet:** {diet_match['Diet'].values[0]}")
                        else:
                            st.write("🥗 **Diet:** Not available")
                        
                        # Workout
                        workout_match = workouts[workouts['Disease'] == prediction]
                        if not workout_match.empty and 'workout' in workout_match.columns:
                            st.write(f"🏋️ **Workout:** {workout_match['workout'].values[0]}")
                        else:
                            st.write("🏋️ **Workout:** Not available")
                            
                    except (IndexError, KeyError) as e:
                        st.write("Treatment plan data is incomplete for this specific condition.")
                        logger.warning(f"Treatment plan error for {prediction}: {e}")
            
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                logger.error(f"Prediction error: {e}")

st.markdown("---")
st.caption("Disclaimer: This is a student project and does not provide professional medical advice.")
