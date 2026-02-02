import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Netflix ML Analyzer",
    page_icon="🎬",
    layout="centered"
)

# -------------------------
# Load Models
# -------------------------
@st.cache_resource
def load_models():
    dt = joblib.load("decision_tree_model.pkl")
    nb = joblib.load("naive_bayes_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return dt, nb, tfidf

dt_model, nb_model, tfidf = load_models()

# -------------------------
# Header
# -------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🎬 Netflix Content Analyzer</h1>
    <p style='text-align:center;color:gray;'>
    Machine Learning based Movie vs TV Show Prediction
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(
    ["🌳 Decision Tree Model", "📄 Naive Bayes Model", "ℹ️ About"]
)

# ==================================================
# 🌳 DECISION TREE TAB
# ==================================================
with tab1:
    st.subheader("🌳 Structured Data Prediction")

    col1, col2 = st.columns(2)

    with col1:
        release_year = st.number_input(
            "📅 Release Year",
            min_value=1900,
            max_value=2025,
            value=2020
        )

    with col2:
        duration = st.number_input(
            "⏱ Duration (minutes)",
            min_value=1,
            max_value=500,
            value=90
        )

    st.markdown("")

    if st.button("🔍 Predict Content Type", use_container_width=True):
        input_df = pd.DataFrame(
            [[release_year, duration]],
            columns=["release_year", "duration"]
        )

        prediction = dt_model.predict(input_df)[0]

        st.markdown("---")
        if prediction == 1:
            st.success("🎥 **Prediction: Movie**")
        else:
            st.info("📺 **Prediction: TV Show**")

# ==================================================
# 📄 NAIVE BAYES TAB
# ==================================================
with tab2:
    st.subheader("📄 Text-Based Prediction (Description)")

    description = st.text_area(
        "📝 Enter Netflix Description",
        placeholder="A thrilling crime drama about power and betrayal...",
        height=120
    )

    if st.button("🧠 Analyze Description", use_container_width=True):
        if description.strip() == "":
            st.warning("⚠️ Please enter a description")
        else:
            vector = tfidf.transform([description])
            prediction = nb_model.predict(vector)[0]

            st.markdown("---")
            if prediction == 1:
                st.success("🎥 **Prediction: Movie**")
            else:
                st.info("📺 **Prediction: TV Show**")

# ==================================================
# ℹ️ ABOUT TAB
# ==================================================
with tab3:
    st.markdown(
        """
        ### 📌 Project Information
        - Dataset: Netflix Titles
        - ML Models:
            - Decision Tree (Structured Data)
            - Naive Bayes (Text Classification)
        - Features:
            - Release Year
            - Duration
            - Description (TF-IDF)
        
        ### 🧠 Why Two Models?
        - Decision Tree → interpretable rules
        - Naive Bayes → strong NLP performance
        
        ### 🚀 Built With
        - Python
        - Scikit-Learn
        - Streamlit
        """
    )

st.divider()
st.caption("© Netflix ML Mini Project | Academic Use")
