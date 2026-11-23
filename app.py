
import streamlit as st
import sqlite3
import json
import pandas as pd

from datetime import datetime

from fake_news_model import predict_news
from toxic_model import predict_toxicity


# DATABASE CONFIGURATION

DB_PATH = "truthguard_history.db"


def create_database():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_history (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            detection_type TEXT,

            input_text TEXT,

            prediction TEXT,

            confidence REAL,

            details TEXT,

            created_at TEXT

        )
    """)

    conn.commit()
    conn.close()


# SAVE HISTORY

def save_history(
    detection_type,
    input_text,
    prediction,
    confidence=0,
    details=None
):

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO detection_history
        (
            detection_type,
            input_text,
            prediction,
            confidence,
            details,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (

        detection_type,
        input_text,
        prediction,
        confidence,
        json.dumps(details or {}),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ))

    conn.commit()
    conn.close()


# GET HISTORY

def get_history():

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        "SELECT * FROM detection_history ORDER BY id DESC",
        conn
    )

    conn.close()

    return df


# CLEAR HISTORY

def clear_history():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("DELETE FROM detection_history")

    conn.commit()
    conn.close()


# PAGE CONFIGURATION

st.set_page_config(
    page_title="TruthGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# CUSTOM CSS

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: white;
}

h1, h2, h3 {
    color: #60a5fa;
}

.subtitle {
    color: #94a3b8;
    font-size: 18px;
}

div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
    height: 45px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# CREATE DATABASE

create_database()


# HEADER

st.title("🛡️ TruthGuard AI")

st.markdown(
    '<p class="subtitle">'
    'AI-Powered Fake News and Toxic Content Detection System'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# SIDEBAR NAVIGATION

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Detection Type",

    [
        "Home",
        "Fake News Detection",
        "Toxic Content Detection",
        "Detection History"
    ]

)


# HOME PAGE

if page == "Home":

    st.header("Welcome to TruthGuard AI")

    st.write(
        "An intelligent platform designed to identify "
        "misinformation and harmful online content."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            "📰 Fake News Detection\n\n"
            "Analyze news articles and identify "
            "potential fake or real news."
        )

    with col2:

        st.warning(
            "🚨 Toxic Content Detection\n\n"
            "Identify harmful comments and "
            "toxic online messages."
        )

    st.divider()

    st.subheader("Our Technology")

    st.write("""
    - Natural Language Processing (NLP)
    - TF-IDF Vectorization
    - Logistic Regression
    - Multi-label Classification
    - Machine Learning
    - SQLite Database
    """)


# FAKE NEWS DETECTION

elif page == "Fake News Detection":

    st.header("📰 Fake News Detection")

    st.write(
        "Enter a news article or headline to analyze its authenticity."
    )

    news_text = st.text_area(
        "Enter News Content",
        height=200,
        placeholder="Paste your news article here..."
    )

    if st.button("Analyze News"):

        if not news_text.strip():

            st.error("Please enter news content.")

        else:

            try:

                with st.spinner("Analyzing news..."):

                    result, confidence = predict_news(news_text)

                if result == "Fake News":

                    st.error("🚨 Prediction: Fake News")

                else:

                    st.success("✅ Prediction: Real News")

                st.metric(
                    "Model Confidence",
                    f"{confidence}%"
                )

                st.progress(
                    min(max(confidence / 100, 0.0), 1.0)
                )

                st.caption(
                    "This is a model prediction, "
                    "not independent verification."
                )

                # SAVE RESULT

                save_history(
                    "Fake News Detection",
                    news_text,
                    result,
                    confidence
                )

                st.success("Analysis saved to Detection History.")

            except Exception as e:

                st.error(f"Prediction failed: {e}")


# TOXIC CONTENT DETECTION

elif page == "Toxic Content Detection":

    st.header("🚨 Toxic Content Detection")

    st.write(
        "Enter a comment or message to identify harmful content."
    )

    comment = st.text_area(
        "Enter Comment",
        height=150,
        placeholder="Type your comment here..."
    )

    if st.button("Analyze Comment"):

        if not comment.strip():

            st.error("Please enter a comment.")

        else:

            try:

                with st.spinner("Analyzing comment..."):

                    results = predict_toxicity(comment)

                toxic_categories = [

                    name for name, value in results.items()

                    if value["prediction"] == 1

                ]

                # OVERALL RESULT

                if toxic_categories:

                    prediction = "Potentially Toxic Content"

                    st.error(
                        "⚠️ Potentially Toxic Content Detected"
                    )

                    st.subheader("Detected Categories")

                    for category in toxic_categories:

                        st.write(
                            f"🔴 {category.replace('_', ' ').title()}"
                        )

                else:

                    prediction = "No Toxic Categories Detected"

                    st.success(
                        "✅ No Toxic Categories Detected"
                    )

                # CATEGORY ANALYSIS

                st.subheader("Category Analysis")

                for category, value in results.items():

                    name = category.replace("_", " ").title()

                    confidence = value["confidence"]

                    prediction_status = value["prediction"]

                    st.write(f"**{name}**")

                    st.progress(
                        min(max(confidence / 100, 0.0), 1.0)
                    )

                    st.caption(
                        f"Prediction: "
                        f"{'Detected' if prediction_status == 1 else 'Not Detected'}"
                        f" | Score: {confidence}%"
                    )

                st.caption(
                    "Scores are model estimates, not verified probabilities."
                )

                # SAVE RESULT

                save_history(
                    "Toxic Content Detection",
                    comment,
                    prediction,
                    max(
                        value["confidence"]
                        for value in results.values()
                    ),
                    results
                )

                st.success("Analysis saved to Detection History.")

            except Exception as e:

                st.error(f"Prediction failed: {e}")


# DETECTION HISTORY

elif page == "Detection History":

    st.header("📊 Detection History")

    st.write(
        "View your previous Fake News and Toxic Content analyses."
    )

    history = get_history()

    if history.empty:

        st.info("No detection history available yet.")

    else:

        # STATISTICS

        total = len(history)

        fake_count = len(
            history[
                history["detection_type"] == "Fake News Detection"
            ]
        )

        toxic_count = len(
            history[
                history["detection_type"] == "Toxic Content Detection"
            ]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Analyses", total)

        col2.metric("News Analyses", fake_count)

        col3.metric("Toxic Analyses", toxic_count)

        st.divider()

        # HISTORY TABLE

        st.subheader("Previous Analyses")

        display_history = history[
            [
                "id",
                "detection_type",
                "prediction",
                "confidence",
                "created_at"
            ]
        ]

        st.dataframe(
            display_history,
            use_container_width=True,
            hide_index=True
        )

        # DETAILED HISTORY

        st.subheader("Analysis Details")

        for _, row in history.iterrows():

            with st.expander(
                f"#{row['id']} - {row['detection_type']} - "
                f"{row['created_at']}"
            ):

                st.write("**Input Content:**")

                st.write(row["input_text"])

                st.write("**Prediction:**")

                st.write(row["prediction"])

                st.write("**Confidence:**")

                st.write(f"{row['confidence']}%")

                if row["details"]:

                    details = json.loads(row["details"])

                    if details:

                        st.write("**Category Details:**")

                        st.json(details)

        st.divider()

        # DOWNLOAD HISTORY

        st.subheader("Download History")

        csv_data = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download History as CSV",
            data=csv_data,
            file_name="truthguard_detection_history.csv",
            mime="text/csv"
        )

        st.divider()

        # CLEAR HISTORY

        st.subheader("Manage History")

        if st.button("🗑️ Clear All History"):

            st.session_state.confirm_clear = True

        if st.session_state.get("confirm_clear", False):

            st.warning(
                "Are you sure you want to delete all detection history?"
            )

            col1, col2 = st.columns(2)

            with col1:

                if st.button("Yes, Delete All"):

                    clear_history()

                    st.session_state.confirm_clear = False

                    st.success("Detection history cleared.")

                    st.rerun()

            with col2:

                if st.button("Cancel"):

                    st.session_state.confirm_clear = False

                    st.rerun()


# FOOTER

st.divider()

st.markdown(
    "<center>Developed as an Open Innovation Project | TruthGuard AI</center>",
    unsafe_allow_html=True
)
