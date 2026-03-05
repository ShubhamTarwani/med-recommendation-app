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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Medical Diagnostic SaaS", layout="wide", page_icon="🩺")

@st.cache_data
def load_and_standardize_data():
    base_path = Path(__file__).parent
    
    files = {
        'train_data': 'Main_Training_2026.csv',
        'description': 'symptom_Description.csv',
        'precaution': 'symptom_precaution.csv',
        'meds': 'medications.csv',
        'diets': 'diets.csv',
        'workouts': 'workout_df.csv'
    }
    
    loaded_data = {}
    
    for key, filename in files.items():
        filepath = base_path / filename
        try:
            if not filepath.exists():
                st.error(f"❌ Error: File '{filename}' not found. Please ensure all data files are in the same directory.")
                logger.error(f"File not found: {filepath}")
                return None, None, None, None, None, None
            
            loaded_data[key] = pd.read_csv(filepath)
            logger.info(f"Successfully loaded {filename}")
            
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

    for df_name, df in [('train_data', train_data), ('description', description), 
                        ('precaution', precaution), ('meds', meds), 
                        ('diets', diets), ('workouts', workouts)]:
        if 'prognosis' in df.columns:
            df.rename(columns={'prognosis': 'Disease'}, inplace=True)
        elif 'disease' in df.columns:
            df.rename(columns={'disease': 'Disease'}, inplace=True)
    
    if 'Disease' not in train_data.columns:
        st.error("❌ Error: 'Disease' column not found in training data after standardization.")
        logger.error("Disease column missing in train_data")
        return None, None, None, None, None, None
    
    return train_data, description, precaution, meds, diets, workouts

train_data, description, precaution, meds, diets, workouts = load_and_standardize_data()

if train_data is None:
    st.stop()

@st.cache_resource
def train_model(data):
    try:
        X = data.select_dtypes(include=[np.number])
        
        if X.empty:
            st.error("❌ Error: No numeric features found in training data.")
            logger.error("No numeric features in training data")
            return None, None, None
        
        if 'Disease' not in data.columns:
            st.error("❌ Error: 'Disease' column not found in training data.")
            logger.error("Disease column missing")
            return None, None, None
        
        y = data['Disease']
        
        if len(X) == 0 or len(y) == 0:
            st.error("❌ Error: Training data is empty.")
            logger.error("Empty training data")
            return None, None, None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced', n_jobs=-1)
        
        with st.spinner("Training model... This may take a moment."):
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model trained successfully. Accuracy: {accuracy:.4f}")
        
        return model, X.columns.values, accuracy
    
    except Exception as e:
        st.error(f"❌ Error during model training: {str(e)}")
        logger.error(f"Model training error: {e}")
        return None, None, None

model, feature_names, model_accuracy = train_model(train_data)

if model is None:
    st.stop()

st.title("🩺 AI-Enabled SaaS Platform: Diagnostic Module")
st.caption("Developed for AIML Coursework - Nagpur 2026")

record_count = len(train_data)
st.markdown(f"**System Status:** Model Trained on {record_count:,} records")
st.divider()

st.sidebar.header("User Symptoms")
selected_symptoms = st.sidebar.multiselect(
    "Select your symptoms:",
    options=[s.replace("_", " ").title() for s in feature_names]
)

if st.sidebar.button("Generate Report"):
    if not selected_symptoms:
        st.sidebar.warning("Please select at least one symptom.")
    else:
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
        
        if np.sum(input_vector) == 0:
            st.error("❌ Error: None of the selected symptoms could be matched. Please try again.")
            logger.warning(f"No symptoms matched from: {selected_symptoms}")
        else:
            if unmatched_symptoms:
                st.sidebar.warning(f"⚠️ Note: {len(unmatched_symptoms)} symptom(s) could not be matched: {', '.join(unmatched_symptoms)}")
            
            try:
                prediction = model.predict([input_vector])[0]
                probabilities = model.predict_proba([input_vector])
                confidence = np.max(probabilities) * 100

                st.subheader(f"Results for: {prediction}")
                st.progress(int(confidence), text=f"AI Confidence Score: {confidence:.2f}%")
                st.write("") 
                
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
                        med_match = meds[meds['Disease'] == prediction]
                        if not med_match.empty and 'Medication' in med_match.columns:
                            st.write(f"💊 **Medication:** {med_match['Medication'].values[0]}")
                        else:
                            st.write("💊 **Medication:** Not available")
                        
                        diet_match = diets[diets['Disease'] == prediction]
                        if not diet_match.empty and 'Diet' in diet_match.columns:
                            st.write(f"🥗 **Diet:** {diet_match['Diet'].values[0]}")
                        else:
                            st.write("🥗 **Diet:** Not available")
                        
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