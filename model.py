"""
SMS Spam Detection using TF-IDF and Machine Learning
=====================================================
Classifies SMS messages as Spam or Ham (legitimate) using:
  - TF-IDF for feature extraction
  - Naive Bayes, Logistic Regression, and SVM classifiers
  - 5-Fold Cross Validation for robust evaluation

Dataset: SMS Spam Collection (Kaggle)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import re
import string
import pickle
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)

# ── Config ──
DATASET = "spam.csv"
OUTPUT_DIR = "outputs"
MODEL_DIR = "saved_models"
RANDOM_STATE = 42
TEST_SIZE = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# =============================================================================
# 1. DATA LOADING
# =============================================================================

print("\n[1/7] Loading dataset...")

df = pd.read_csv(DATASET, encoding="latin-1")
df = df[["v1", "v2"]].copy()
df.columns = ["label", "message"]
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# Engineer some basic features for EDA
df["msg_length"] = df["message"].str.len()
df["word_count"] = df["message"].str.split().str.len()
df["digit_count"] = df["message"].apply(lambda x: sum(c.isdigit() for c in x))

print(f"  Total messages : {len(df)}")
print(f"  Ham            : {(df['label']=='ham').sum()}")
print(f"  Spam           : {(df['label']=='spam').sum()}")
print(f"  Duplicates     : {df.duplicated(subset='message').sum()}")


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================

print("\n[2/7] Running EDA & generating plots...")

# --- 2a. Class distribution ---
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["label"].value_counts()
colors = ["#2ecc71", "#e74c3c"]
bars = ax.bar(["Ham", "Spam"], counts.values, color=colors, edgecolor="black", width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            str(val), ha="center", fontweight="bold")
ax.set_title("Class Distribution", fontweight="bold")
ax.set_ylabel("Count")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/01_class_distribution.png", dpi=150)
plt.close()

# --- 2b. Message length by class ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for label, color in zip(["ham", "spam"], colors):
    subset = df[df["label"] == label]["msg_length"]
    axes[0].hist(subset, bins=50, alpha=0.6, color=color, label=label.title(), edgecolor="white")
axes[0].set_title("Message Length Distribution", fontweight="bold")
axes[0].set_xlabel("Characters")
axes[0].set_ylabel("Frequency")
axes[0].legend()

data = [df[df["label"] == "ham"]["msg_length"], df[df["label"] == "spam"]["msg_length"]]
bp = axes[1].boxplot(data, tick_labels=["Ham", "Spam"], patch_artist=True)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_title("Message Length (Box Plot)", fontweight="bold")
axes[1].set_ylabel("Characters")

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/02_message_length.png", dpi=150)
plt.close()

# --- 2c. Word clouds ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, label, cmap, title in [
    (axes[0], "ham", "Greens", "Ham Messages"),
    (axes[1], "spam", "Reds", "Spam Messages"),
]:
    text = " ".join(df[df["label"] == label]["message"])
    wc = WordCloud(width=700, height=350, background_color="white",
                   colormap=cmap, max_words=100).generate(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.axis("off")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/03_wordclouds.png", dpi=150)
plt.close()

# --- 2d. Correlation heatmap ---
fig, ax = plt.subplots(figsize=(6, 5))
corr_cols = ["msg_length", "word_count", "digit_count", "label_num"]
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax, linewidths=0.5)
ax.set_title("Feature Correlation", fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/04_correlation.png", dpi=150)
plt.close()

# Print key EDA insights
ham_avg = df[df["label"] == "ham"]["msg_length"].mean()
spam_avg = df[df["label"] == "spam"]["msg_length"].mean()
print(f"  Avg ham length  : {ham_avg:.0f} chars")
print(f"  Avg spam length : {spam_avg:.0f} chars  ({spam_avg/ham_avg:.1f}x longer)")
print(f"  Saved 4 EDA plots to {OUTPUT_DIR}/")


# =============================================================================
# 3. TEXT PREPROCESSING
# =============================================================================

print("\n[3/7] Preprocessing text...")

def clean_text(text):
    """Lowercase, remove URLs, digits, punctuation, extra whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)       # urls
    text = re.sub(r"\d+", " ", text)                   # digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_message"] = df["message"].apply(clean_text)

# Show a few examples
print("  Sample:")
for i in [0, 3, 5]:
    print(f"    Original : {df.iloc[i]['message'][:70]}...")
    print(f"    Cleaned  : {df.iloc[i]['clean_message'][:70]}...")
    print()


# =============================================================================
# 4. TRAIN-TEST SPLIT & TF-IDF
# =============================================================================

print("[4/7] Splitting data & extracting TF-IDF features...")

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_message"], df["label_num"],
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label_num"]
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2),    # unigrams + bigrams for better context
    sublinear_tf=True,     # dampens term frequency
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"  Train: {X_train_tfidf.shape[0]} samples | Test: {X_test_tfidf.shape[0]} samples")
print(f"  Vocabulary size: {len(vectorizer.vocabulary_)} features")


# =============================================================================
# 5. MODEL TRAINING & EVALUATION
# =============================================================================

print("\n[5/7] Training models with 5-Fold Cross Validation...\n")

models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, solver="liblinear"),
    "SVM (Linear)": LinearSVC(C=1.0, max_iter=2000, class_weight="balanced"),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}

for name, model in models.items():
    # Cross validation
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=cv, scoring="f1")

    # Train on full training set and predict
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    # Get scores for ROC curve
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_tfidf)[:, 1]
    else:
        y_score = model.decision_function(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)

    results[name] = {
        "model": model, "y_pred": y_pred, "y_score": y_score,
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "auc": auc, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
    }

    print(f"  --- {name} ---")
    print(f"  CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"], digits=4))


# =============================================================================
# 6. EVALUATION PLOTS
# =============================================================================

print("[6/7] Generating evaluation plots...")

# --- 6a. Confusion matrices ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name, fontweight="bold")
fig.suptitle("Confusion Matrices", fontweight="bold", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/05_confusion_matrices.png", dpi=150)
plt.close()

# --- 6b. ROC curves ---
fig, ax = plt.subplots(figsize=(7, 5))
line_colors = ["#2ecc71", "#3498db", "#e74c3c"]
for (name, res), color in zip(results.items(), line_colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_score"])
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves", fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/06_roc_curves.png", dpi=150)
plt.close()

# --- 6c. Model comparison bar chart ---
fig, ax = plt.subplots(figsize=(10, 5))
metrics = ["accuracy", "precision", "recall", "f1", "auc"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
x = np.arange(len(metrics))
width = 0.25

for i, (name, color) in enumerate(zip(results.keys(), line_colors)):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name, color=color, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x + width)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0.6, 1.05)
ax.set_title("Model Comparison", fontweight="bold")
ax.legend()
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/07_model_comparison.png", dpi=150)
plt.close()

# --- 6d. Top predictive words (from Logistic Regression coefficients) ---
lr_model = results["Logistic Regression"]["model"]
feature_names = vectorizer.get_feature_names_out()
coefs = lr_model.coef_[0]

top_n = 15
top_spam = np.argsort(coefs)[-top_n:][::-1]
top_ham = np.argsort(coefs)[:top_n]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].barh(range(top_n), coefs[top_spam], color="#e74c3c")
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(feature_names[top_spam])
axes[0].invert_yaxis()
axes[0].set_title("Top Spam Indicators", fontweight="bold")
axes[0].set_xlabel("Coefficient Weight")

axes[1].barh(range(top_n), abs(coefs[top_ham]), color="#2ecc71")
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels(feature_names[top_ham])
axes[1].invert_yaxis()
axes[1].set_title("Top Ham Indicators", fontweight="bold")
axes[1].set_xlabel("Coefficient Weight (abs)")

plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/08_feature_importance.png", dpi=150)
plt.close()

print(f"  Saved 4 evaluation plots to {OUTPUT_DIR}/")


# =============================================================================
# 7. SAVE BEST MODEL & DEMO PREDICTIONS
# =============================================================================

print("\n[7/7] Saving best model & running demo predictions...")

# Pick best model by F1 score
best_name = max(results, key=lambda k: results[k]["f1"])
best_model = results[best_name]["model"]

with open(f"{MODEL_DIR}/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open(f"{MODEL_DIR}/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print(f"  Best model: {best_name} (F1={results[best_name]['f1']:.4f})")
print(f"  Saved to {MODEL_DIR}/")

# Demo predictions on new messages
demo_messages = [
    "Hey, are we still meeting for lunch tomorrow?",
    "WINNER! You've been selected for a £1000 cash prize! Call NOW!",
    "Can you pick up some milk on your way home?",
    "FREE entry to win an iPhone! Text WIN to 80808",
    "I'll be there in 10 minutes, stuck in traffic.",
    "Urgent! Claim your reward now. Call 0800-123-456",
]

print("\n  --- Demo Predictions ---")
for msg in demo_messages:
    cleaned = clean_text(msg)
    vec = vectorizer.transform([cleaned])
    pred = best_model.predict(vec)[0]
    tag = "SPAM" if pred == 1 else "HAM "
    print(f"  [{tag}] {msg[:65]}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  {'Model':<25s} {'Accuracy':>9s} {'F1':>8s} {'AUC':>8s}")
print(f"  {'-'*25} {'-'*9} {'-'*8} {'-'*8}")
for name, res in results.items():
    marker = " <-- best" if name == best_name else ""
    print(f"  {name:<25s} {res['accuracy']:>8.4f}  {res['f1']:>7.4f}  {res['auc']:>7.4f}{marker}")
print(f"\n  Winner: {best_name}")
print(f"  All outputs saved to: {OUTPUT_DIR}/ and {MODEL_DIR}/")
print("=" * 60)
