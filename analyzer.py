from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
##from sklearn.linear_model import LogisticRegression
##from sklearn.pipeline import Pipeline
##from sklearn.model_selection import train_test_split
##from sklearn.metrics import accuracy_score, classification_report
import re
import pandas as pd
##import pickle
import joblib
import os

analyzer = SentimentIntensityAnalyzer()

# ── Text cleaning ─────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r'http\S+', '', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip().lower()

# ── VADER sentiment ───────────────────────────────────────────
def get_sentiment(text):
    scores  = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label, emoji, color = "Positive", "😊", "#22c55e"
    elif compound <= -0.05:
        label, emoji, color = "Negative", "😠", "#ef4444"
    else:
        label, emoji, color = "Neutral",  "😐", "#f59e0b"
    return {
        "label": label, "emoji": emoji, "color": color,
        "compound":    round(compound, 4),
        "positive":    round(scores["pos"], 4),
        "negative":    round(scores["neg"], 4),
        "neutral":     round(scores["neu"], 4),
        "confidence":  round(abs(compound) * 100, 1)
    }

# ── TextBlob sentiment ────────────────────────────────────────
def get_textblob_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity >= 0.05:
        label, color = "Positive", "#22c55e"
    elif polarity <= -0.05:
        label, color = "Negative", "#ef4444"
    else:
        label, color = "Neutral",  "#f59e0b"
    return {
        "label":      label,
        "color":      color,
        "score":      round(polarity, 4),
        "confidence": round(abs(polarity) * 100, 1)
    }

# ── Logistic Regression model ─────────────────────────────────
MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


def load_lr_model():
    if not os.path.exists(MODEL_PATH):
        return None, None

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def get_lr_sentiment(text, model, vectorizer):

    if model is None or vectorizer is None:
        return None

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    confidence = round(max(probability) * 100, 2)

    sentiment = "Positive" if prediction == 1 else "Negative"

    color = {
        "Positive": "#22c55e",
        "Negative": "#ef4444"
    }

    return {
        "label": sentiment,
        "confidence": confidence,
        "color": color[sentiment],
        "probabilities": probability
    }
# ── Keywords ──────────────────────────────────────────────────
def get_keywords(text, top_n=7):
    cleaned = clean_text(text)
    if len(cleaned.split()) < 3:
        return []
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=50)
        vectorizer.fit_transform([cleaned])
        scores_map = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
        sorted_kw  = sorted(scores_map.items(), key=lambda x: x[1])
        result = []
        for kw, _ in sorted_kw[:top_n]:
            s = analyzer.polarity_scores(kw)["compound"]
            result.append((kw, "pos" if s>=0.05 else ("neg" if s<=-0.05 else "neu")))
        return result
    except:
        return []

# ── Fast bulk analysis ────────────────────────────────────────
def analyze_bulk(reviews):
    series     = pd.Series([str(r) for r in reviews])
    scores     = series.apply(lambda x: analyzer.polarity_scores(x))
    compounds  = scores.apply(lambda s: round(s["compound"], 4))
    positives  = scores.apply(lambda s: round(s["pos"], 4))
    negatives  = scores.apply(lambda s: round(s["neg"], 4))
    neutrals   = scores.apply(lambda s: round(s["neu"], 4))
    labels     = compounds.apply(lambda c: "Positive" if c>=0.05 else ("Negative" if c<=-0.05 else "Neutral"))
    confidences= compounds.apply(lambda c: f"{round(abs(c)*100,1)}%")
    short_text = series.apply(lambda t: t[:80]+"..." if len(t)>80 else t)
    return pd.DataFrame({
        "Review":     short_text.values,
        "Sentiment":  labels.values,
        "Score":      compounds.values,
        "Positive":   positives.values,
        "Negative":   negatives.values,
        "Neutral":    neutrals.values,
        "Confidence": confidences.values,
    })


import numpy as np

def explain_prediction(model, vectorizer, top_n=5):
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    top_positive = np.argsort(coefficients)[-top_n:][::-1]
    top_negative = np.argsort(coefficients)[:top_n]

    positive_words = [
        (feature_names[i], round(coefficients[i], 2))
        for i in top_positive
    ]

    negative_words = [
        (feature_names[i], round(coefficients[i], 2))
        for i in top_negative
    ]

    return positive_words, negative_words
