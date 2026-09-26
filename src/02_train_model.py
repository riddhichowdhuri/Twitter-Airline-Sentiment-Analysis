"""
02_train_model.py
Trains a simple, interpretable sentiment classifier on the airline
tweets: TF-IDF features + Logistic Regression (multi-class: negative /
neutral / positive).

Produces (into ../output/):
  - model_metrics.txt     : accuracy, classification report
  - confusion_matrix.png  : confusion matrix heatmap
  - top_features.png      : words most associated with each sentiment
  - predictions_sample.csv: a few held-out tweets with true vs predicted label

Run from the project root (after 01_eda.py, though it does not depend on it):
    python src/02_train_model.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(os.path.dirname(__file__))
from clean_text import clean_tweet  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Tweets.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUT_DIR, exist_ok=True)

LABELS = ["negative", "neutral", "positive"]
RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["text"].apply(clean_tweet)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["airline_sentiment"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["airline_sentiment"],
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=LABELS)

    metrics_lines = [
        "Model: TF-IDF (unigrams+bigrams, 5000 features) + Logistic Regression",
        f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}",
        f"Accuracy: {acc:.4f}",
        "",
        "Classification report:",
        report,
    ]
    metrics_path = os.path.join(OUT_DIR, "model_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("\n".join(metrics_lines))
    print("\n".join(metrics_lines))
    print(f"Saved: {metrics_path}")

    # --- Confusion matrix ---------------------------------------------------
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (accuracy = {acc:.1%})")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # --- Top words per class -------------------------------------------------
    feature_names = np.array(vectorizer.get_feature_names_out())
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for idx, label in enumerate(LABELS):
        class_idx = list(model.classes_).index(label)
        coefs = model.coef_[class_idx]
        top_idx = np.argsort(coefs)[-12:]
        axes[idx].barh(feature_names[top_idx], coefs[top_idx], color="#1f77b4")
        axes[idx].set_title(f"Top words -> '{label}'")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "top_features.png"), dpi=150)
    plt.close()

    # --- Save a small sample of predictions for a quick sanity check --------
    sample = pd.DataFrame({
        "tweet": X_test,
        "true_sentiment": y_test,
        "predicted_sentiment": y_pred,
    }).sample(15, random_state=RANDOM_STATE)
    sample.to_csv(os.path.join(OUT_DIR, "predictions_sample.csv"), index=False)

    print("Saved confusion_matrix.png, top_features.png, predictions_sample.csv to", OUT_DIR)


if __name__ == "__main__":
    main()
