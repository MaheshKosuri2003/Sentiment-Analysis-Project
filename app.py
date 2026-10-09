import re
import warnings
from pathlib import Path

import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Keep app.py and these three files in the same directory.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgboost_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    page_icon="💬",
    layout="centered",
)


@st.cache_resource
def load_nltk_resources():
    """Download required NLTK data if it is not already installed."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=True)

    stop_words = set(stopwords.words("english"))
    # Do not remove negation; it changes the meaning of a review.
    stop_words -= {"no", "not", "never", "neither", "dont", "nor"}
    return stop_words, WordNetLemmatizer()


@st.cache_resource
def load_artifacts():
    """Load the best model selected from the notebook's saved evaluation."""
    for path in (MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path.name}. "
                "Place it in the same folder as app.py."
            )

    # Suppress only compatibility warnings during loading; see note below.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        label_encoder = joblib.load(ENCODER_PATH)

    stop_words, lemmatizer = load_nltk_resources()
    return model, vectorizer, label_encoder, stop_words, lemmatizer


def clean_text(text):
    """Match the notebook's first cleaning step."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text, stop_words, lemmatizer):
    """Match the notebook's stop-word filtering and lemmatization."""
    text = clean_text(text)
    tokens = word_tokenize(text)

    spec_pattern = re.compile(
        r"^\d+(mAh|hz|mp|gb|tb|w|inch|ghz|nm|fps|ppi|nits)$",
        re.IGNORECASE,
    )
    cleaned_tokens = []

    for word in tokens:
        word_lower = word.lower()

        if word.isalpha() and word_lower not in stop_words:
            cleaned_tokens.append(lemmatizer.lemmatize(word_lower))
        elif spec_pattern.match(word_lower):
            cleaned_tokens.append(word_lower)
        elif word.isdigit() and 1 <= int(word) <= 10:
            cleaned_tokens.append(word_lower)

    return " ".join(cleaned_tokens)


def sentiment_style(label):
    label = str(label).strip().lower()
    if label == "positive":
        return "success"
    if label == "negative":
        return "error"
    return "info"


def predict_sentiment(review, model, vectorizer, encoder, stop_words, lemmatizer):
    processed = preprocess_text(review, stop_words, lemmatizer)
    if not processed:
        raise ValueError(
            "No usable words remained after preprocessing. "
            "Please enter a longer review."
        )

    features = vectorizer.transform([processed])
    prediction_id = int(model.predict(features)[0])
    sentiment = str(encoder.inverse_transform([prediction_id])[0])

    probability = None
    if hasattr(model, "predict_proba"):
        try:
            class_probabilities = model.predict_proba(features)[0]
            probability = float(class_probabilities.max())
        except (ValueError, AttributeError):
            probability = None

    return sentiment, processed, probability


st.title("💬 Amazon Review Sentiment Analysis")
st.write(
    "Analyze a customer review using the **XGBoost classifier**. "
    "The model predicts Positive, Neutral, or Negative sentiment."
)

st.caption(
    "Selected model: XGBoost — highest weighted F1-score among the five "
    "models evaluated in the supplied notebook."
)

try:
    model, vectorizer, encoder, stop_words, lemmatizer = load_artifacts()
except Exception as exc:
    st.error("The application could not load its model files or text resources.")
    st.exception(exc)
    st.stop()

review = st.text_area(
    "Customer review",
    placeholder=(
        "Example: The battery lasts all day and the camera quality is excellent."
    ),
    height=160,
)

if st.button("Analyze sentiment", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a customer review first.")
    else:
        try:
            sentiment, processed, probability = predict_sentiment(
                review, model, vectorizer, encoder, stop_words, lemmatizer
            )

            style = sentiment_style(sentiment)
            if style == "success":
                st.success(f"Predicted sentiment: **{sentiment}** 😊")
            elif style == "error":
                st.error(f"Predicted sentiment: **{sentiment}** 😞")
            else:
                st.info(f"Predicted sentiment: **{sentiment}** 😐")

            if probability is not None:
                st.metric("Maximum predicted class probability", f"{probability:.1%}")
                st.caption(
                    "This is the model's estimated probability for its most likely "
                    "class, not a guarantee of correctness."
                )

            with st.expander("View preprocessed text"):
                st.write(processed)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.divider()
st.caption(
    "Evaluation note: the notebook reports a weighted F1-score of about 0.780 "
    "for XGBoost. Neutral reviews were harder to classify, so predictions for "
    "that class may be less reliable."
)
