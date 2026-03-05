import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- UI Configuration ---
st.set_page_config(page_title="Medicine Marketplace SaaS", layout="wide", page_icon="💊")

# --- Data Loading & AI Training (Cached for Speed) ---
@st.cache_data
def load_and_train_ai():
    """Loads 1mg clinical data and trains the NLP model."""
    df_1mg = pd.read_csv('1mg.csv')
    df_1mg['desc'] = df_1mg['desc'].fillna('')
    df_1mg['activeIngredient'] = df_1mg['activeIngredient'].fillna('')
    
    # Train the NLP Vectorizer on the medical descriptions
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df_1mg['desc'])
    
    return df_1mg, vectorizer, tfidf_matrix

@st.cache_data
def load_marketplace_inventory():
    """Loads and cleans the A-Z Marketplace data."""
    df_az = pd.read_csv('A_Z_medicines_dataset_of_India.csv')
    
    # Clean the price column: extract numbers and convert to float
    df_az['clean_price'] = df_az['price(₹)'].replace(r'[^\d.]', '', regex=True)
    df_az['clean_price'] = pd.to_numeric(df_az['clean_price'], errors='coerce').fillna(0)
    
    # Ensure compositions are searchable strings
    df_az['short_composition1'] = df_az['short_composition1'].fillna('').astype(str)
    df_az['short_composition2'] = df_az['short_composition2'].fillna('').astype(str)
    
    # Handle the boolean/string nature of the Is_discontinued column
    df_az['Is_discontinued'] = df_az['Is_discontinued'].astype(str).str.upper() == 'TRUE'
    
    return df_az

# Initialize the backend
with st.spinner("Initializing AI Engine and loading marketplace inventory..."):
    df_1mg, vectorizer, tfidf_matrix = load_and_train_ai()
    df_az = load_marketplace_inventory()

# --- Main App Layout ---
st.title("💊 AI Enabled SaaS Platform for Medicine Marketplace")
st.markdown("Enter patient symptoms in plain English. The AI will cross-reference clinical profiles and find the most affordable commercial options currently available.")

# --- Sidebar ---
st.sidebar.header("⚙️ Platform Engine Stats")
st.sidebar.metric(label="Clinical Profiles Analyzed", value=f"{len(df_1mg):,}")
st.sidebar.metric(label="Commercial Inventory Size", value=f"{len(df_az):,}")
st.sidebar.divider()
st.sidebar.info("Model: TF-IDF + Cosine Similarity\n\nRouting delivery via Pimpri-Chinchwad distribution nodes.")

# --- Application Logic ---
st.subheader("1. Symptom Assessment")
user_symptoms = st.text_area("Describe the condition (e.g., 'severe nerve pain, depression, and anxiety'):", height=100)

if st.button("Diagnose & Find Medicines", type="primary"):
    if user_symptoms.strip() == "":
        st.error("Please enter symptoms to begin the analysis.")
    else:
        # --- Phase 1: AI Clinical Matching ---
        user_tfidf = vectorizer.transform([user_symptoms])
        cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
        best_match_idx = cosine_similarities.argmax()
        confidence_score = cosine_similarities[best_match_idx] * 100
        
        best_match_drug = df_1mg.iloc[best_match_idx]
        predicted_ingredient = best_match_drug['activeIngredient']
        
        st.divider()
        st.subheader("2. AI Clinical Results")
        
        if confidence_score < 1.0:
            st.warning("⚠️ **Low Confidence Match:** The AI could not find a strong clinical correlation. Please provide more detailed symptoms.")
        else:
            st.success(f"**AI Match Found:** {best_match_drug['name']} (Match Confidence: {confidence_score:.2f}%)")
            st.markdown(f"**🔬 Predicted Active Ingredient/Salt:** `{predicted_ingredient}`")
            
            # Display Safety Warnings
            st.markdown("#### ⚠️ Clinical Safety Warnings")
            col1, col2, col3 = st.columns(3)
            col1.info(f"**🍷 Alcohol:** {best_match_drug['alcoholWarning']}")
            col2.warning(f"**🤰 Pregnancy:** {best_match_drug['pregnancyWarning']}")
            col3.success(f"**🤱 Breastfeeding:** {best_match_drug['breastfeedingWarning']}")
            
            # --- Phase 2: Marketplace Search ---
            st.divider()
            st.subheader(f"3. Marketplace: Cheapest Options for '{predicted_ingredient}'")
            
            if not predicted_ingredient:
                st.error("No active ingredient found in the knowledge base for this match.")
            else:
                # Extract the main chemical name (e.g., grab "Duloxetine" from "Duloxetine (30mg)")
                main_chemical = predicted_ingredient.split()[0].strip()
                
                # Search the A-Z inventory in both composition columns
                matches = df_az[
                    df_az['short_composition1'].str.contains(main_chemical, case=False, na=False) |
                    df_az['short_composition2'].str.contains(main_chemical, case=False, na=False)
                ]
                
                # Filter out discontinued drugs
                matches = matches[~matches['Is_discontinued']]
                
                if not matches.empty:
                    # Sort by cheapest price
                    cheapest_options = matches.sort_values(by='clean_price').head(10)
                    
                    # Format the output dataframe
                    display_df = cheapest_options[['name', 'manufacturer_name', 'pack_size_label', 'price(₹)']].copy()
                    display_df.columns = ['Brand Name', 'Manufacturer', 'Pack Size', 'Price (₹)']
                    
                    # Display the interactive dataframe
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    st.caption(f"Showing the top {len(display_df)} most affordable commercial alternatives currently available.")
                else:
                    st.warning(f"No commercial alternatives found in the current marketplace inventory for `{main_chemical}`.")