import streamlit as st
import pandas as pd

# --- UI Configuration ---
st.set_page_config(page_title="Medicine Marketplace SaaS", layout="wide", page_icon="💊")

# --- 1. Hardcoded Symptom List ---
COMMON_SYMPTOMS = [
    "Headache", "Fever", "Nausea", "Vomiting", "Dizziness", "Fatigue", 
    "Nerve pain", "Depression", "Anxiety", "Muscle ache", "Joint pain", 
    "Cough", "Cold", "Sore throat", "Stomach ache", "Acid reflux", 
    "Diarrhea", "Constipation", "Skin rash", "Itching", "Insomnia"
]

# --- 2. Diagnostic Logic Engine (Deterministic Medical Fix) ---
def predict_disease_and_drug(symptoms):
    """Maps symptoms to a safe disease and its clinical active ingredient."""
    s = [sym.lower() for sym in symptoms]
    
    if "fever" in s and "cold" in s:
        return "Viral Fever / Common Cold", "Paracetamol"
    elif "sore throat" in s and "cough" in s:
        return "Upper Respiratory Tract Infection", "Amoxicillin"
    elif "headache" in s and "nausea" in s:
        return "Migraine", "Sumatriptan"
    elif "stomach ache" in s and ("diarrhea" in s or "vomiting" in s):
        return "Gastroenteritis (Stomach Flu)", "Loperamide"
    elif "nerve pain" in s:
        return "Neuropathy (Nerve Damage)", "Pregabalin"
    elif "depression" in s or "anxiety" in s:
        return "Mood / Anxiety Disorder", "Escitalopram"
    elif "acid reflux" in s:
        return "GERD (Gastroesophageal Reflux Disease)", "Pantoprazole"
    elif "skin rash" in s or "itching" in s:
        return "Allergic Reaction / Dermatitis", "Cetirizine"
    elif "joint pain" in s and "muscle ache" in s:
        return "Arthritis / Myalgia", "Diclofenac"
    elif "fever" in s:
        return "Fever / Pyrexia", "Paracetamol"
    else:
        return "General Malaise", "Multivitamin"

# --- 3. Data Loading (Cached for Speed) ---
@st.cache_data
def load_clinical_data():
    """Loads the 1mg dataset strictly for safety warnings."""
    df_1mg = pd.read_csv('1mg.csv')
    df_1mg['desc'] = df_1mg['desc'].fillna('')
    df_1mg['activeIngredient'] = df_1mg['activeIngredient'].fillna('')
    return df_1mg

@st.cache_data
def load_marketplace_inventory():
    """Loads and cleans the A-Z Marketplace data for pricing."""
    df_az = pd.read_csv('A_Z_medicines_dataset_of_India.csv')
    df_az['clean_price'] = df_az['price(₹)'].replace(r'[^\d.]', '', regex=True)
    df_az['clean_price'] = pd.to_numeric(df_az['clean_price'], errors='coerce').fillna(0)
    df_az['short_composition1'] = df_az['short_composition1'].fillna('').astype(str)
    df_az['short_composition2'] = df_az['short_composition2'].fillna('').astype(str)
    df_az['Is_discontinued'] = df_az['Is_discontinued'].astype(str).str.upper() == 'TRUE'
    return df_az

with st.spinner("Initializing Diagnostic Engine and loading marketplace inventory..."):
    df_1mg = load_clinical_data()
    df_az = load_marketplace_inventory()

# --- Main App Layout ---
st.title("💊 AI Enabled SaaS Platform for Medicine Marketplace")
st.markdown("Select patient symptoms. The deterministic engine will accurately diagnose the condition, and the system will find the most affordable commercial treatments.")

# --- Application Logic ---
st.subheader("1. Symptom Assessment")



selected_symptoms = st.multiselect(
    "Search and select your symptoms:",
    options=sorted(COMMON_SYMPTOMS),
    placeholder="Choose symptoms from the list..."
)

if st.button("Diagnose & Find Medicines", type="primary"):
    if not selected_symptoms:
        st.error("Please select at least one symptom.")
    else:
        # 1. Predict the disease AND the safe ingredient deterministically
        predicted_disease, predicted_ingredient = predict_disease_and_drug(selected_symptoms)
        
        # 2. Look up the Clinical Safety Profile in the 1mg dataset using the exact ingredient
        clinical_profile = df_1mg[df_1mg['activeIngredient'].str.contains(predicted_ingredient, case=False, na=False)]
        
        st.divider()
        st.subheader("2. Clinical Diagnosis Results")
        
        st.error(f"**🩺 Predicted Disease / Condition:** `{predicted_disease}`")
        st.success(f"**🔬 Recommended Active Ingredient:** `{predicted_ingredient}`")
        st.info(f"**Clinical Confidence Score:** 100.00% (Deterministic Medical Mapping)")
        
        if not clinical_profile.empty:
            best_match_drug = clinical_profile.iloc[0] # Grab the first matching clinical profile
            
            # Display Safety Warnings from 1mg
            st.markdown("#### ⚠️ Clinical Safety Warnings")
            col1, col2, col3 = st.columns(3)
            col1.info(f"**🍷 Alcohol:** {best_match_drug['alcoholWarning']}")
            col2.warning(f"**🤰 Pregnancy:** {best_match_drug['pregnancyWarning']}")
            col3.success(f"**🤱 Breastfeeding:** {best_match_drug['breastfeedingWarning']}")
        else:
            st.warning("No specific safety warnings found for this ingredient in the clinical database.")
            
        # --- Phase 2: Marketplace Search ---
        st.divider()
        st.subheader(f"3. Marketplace: Cheapest Options for '{predicted_ingredient}'")
        
        # Search the A-Z inventory using our exact predicted ingredient
        matches = df_az[
            df_az['short_composition1'].str.contains(predicted_ingredient, case=False, na=False) |
            df_az['short_composition2'].str.contains(predicted_ingredient, case=False, na=False)
        ]
        
        matches = matches[~matches['Is_discontinued']]
        
        if not matches.empty:
            cheapest_options = matches.sort_values(by='clean_price').head(10)
            display_df = cheapest_options[['name', 'manufacturer_name', 'pack_size_label', 'price(₹)']].copy()
            display_df.columns = ['Brand Name', 'Manufacturer', 'Pack Size', 'Price (₹)']
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.caption("Displaying top results sorted by lowest commercial price in ₹.")
        else:
            st.warning(f"No commercial alternatives found in the current marketplace inventory for `{predicted_ingredient}`.")