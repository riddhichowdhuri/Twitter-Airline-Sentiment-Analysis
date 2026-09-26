"""
01_eda.py
Exploratory analysis of the Twitter US Airline Sentiment dataset.

Produces (into ../output/):
  - eda_summary.txt        : plain-text summary stats
  - sentiment_counts.png   : overall sentiment distribution
  - sentiment_by_airline.png : sentiment split per airline
  - negative_reasons.png   : top reasons behind negative tweets
  - tweet_length_hist.png  : distribution of tweet length by sentiment

Run from the project root:
    python src/01_eda.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Tweets.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUT_DIR, exist_ok=True)

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}


def main():
    df = pd.read_csv(DATA_PATH)

    lines = []
    lines.append(f"Total tweets: {len(df)}")
    lines.append(f"Airlines covered: {', '.join(sorted(df['airline'].unique()))}")
    lines.append("")
    lines.append("Overall sentiment counts:")
    counts = df["airline_sentiment"].value_counts().reindex(SENTIMENT_ORDER)
    for label, n in counts.items():
        lines.append(f"  {label:>9}: {n} ({n / len(df):.1%})")
    lines.append("")
    lines.append("Top 10 negative reasons:")
    top_reasons = df["negativereason"].value_counts().head(10)
    for reason, n in top_reasons.items():
        lines.append(f"  {reason}: {n}")

    summary_path = os.path.join(OUT_DIR, "eda_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved: {summary_path}")

    # --- Chart 1: overall sentiment distribution -------------------------
    plt.figure(figsize=(5, 4))
    counts.plot(kind="bar", color=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER])
    plt.title("Overall sentiment distribution")
    plt.ylabel("Number of tweets")
    plt.xlabel("Sentiment")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sentiment_counts.png"), dpi=150)
    plt.close()

    # --- Chart 2: sentiment by airline ------------------------------------
    pivot = (
        df.groupby(["airline", "airline_sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SENTIMENT_ORDER)
    )
    pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(7, 5),
        color=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER],
    )
    plt.title("Sentiment by airline")
    plt.ylabel("Number of tweets")
    plt.xlabel("Airline")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sentiment_by_airline.png"), dpi=150)
    plt.close()

    # --- Chart 3: top negative reasons ------------------------------------
    plt.figure(figsize=(7, 5))
    top_reasons.sort_values().plot(kind="barh", color="#d62728")
    plt.title("Top reasons behind negative tweets")
    plt.xlabel("Number of tweets")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "negative_reasons.png"), dpi=150)
    plt.close()

    # --- Chart 4: tweet length distribution by sentiment -------------------
    df["tweet_length"] = df["text"].str.len()
    plt.figure(figsize=(6, 4))
    for sentiment in SENTIMENT_ORDER:
        subset = df.loc[df["airline_sentiment"] == sentiment, "tweet_length"]
        plt.hist(subset, bins=30, alpha=0.5, label=sentiment, color=SENTIMENT_COLORS[sentiment])
    plt.title("Tweet length by sentiment")
    plt.xlabel("Character count")
    plt.ylabel("Number of tweets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tweet_length_hist.png"), dpi=150)
    plt.close()

    print("Saved 4 charts to", OUT_DIR)


if __name__ == "__main__":
    main()
