import streamlit as st

from fake_news_model import predict_news
from toxic_model import predict_toxicity


# Page configuration
st.set_page_config(
    page_title="TruthGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# Custom CSS
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

.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #172554;
    margin-top: 15px;
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


# Header
st.title("🛡️ TruthGuard AI")

st.markdown(
    '<p class="subtitle">AI-Powered Fake News and Toxic Content Detection System</p>',
    unsafe_allow_html=True
)

st.divider()


# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Detection Type",
    [
        "Home",
        "Fake News Detection",
        "Toxic Content Detection"
    ]
)


# Home page
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
    """)


# Fake News Detection
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
                    "This is a model prediction, not independent verification."
                )

            except Exception as e:

                st.error(f"Prediction failed: {e}")


# Toxic Content Detection
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

                if toxic_categories:

                    st.error("⚠️ Potentially Toxic Content Detected")

                    st.subheader("Detected Categories")

                    for category in toxic_categories:

                        st.write(
                            f"🔴 {category.replace('_', ' ').title()}"
                        )

                else:

                    st.success("✅ No Toxic Categories Detected")

                st.subheader("Category Analysis")

                for category, value in results.items():

                    name = category.replace("_", " ").title()

                    st.write(f"**{name}**")

                    st.progress(
                        min(
                            max(value["confidence"] / 100, 0.0),
                            1.0
                        )
                    )

                    st.caption(
                        f"Prediction: "
                        f"{'Detected' if value['prediction'] == 1 else 'Not Detected'}"
                        f" | Score: {value['confidence']}%"
                    )

            except Exception as e:

                st.error(f"Prediction failed: {e}")


# Footer
st.divider()

st.markdown(
    "<center>Developed as an Open Innovation Project | TruthGuard AI</center>",
    unsafe_allow_html=True
)