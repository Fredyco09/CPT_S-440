import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from preprocess import preprocess_batch

LABEL_MAP = {0: "Negative", 1: "Positive"}

def plot_confusion_matrix(cm, output_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title("Naive Bayes Confusion Matrix (SST-2)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix_nb.png")
    plt.savefig(path)
    plt.close()
    print(f"Confusion matrix saved to {path}")


def error_analysis(X_test_raw, y_test, y_pred, output_dir, n=10):
    """ This prints misclassified examples and plots error category breakdown
        Categories:
            - Sarcasm: positive-sounding structure with negative intent
            - Implicit Negativity: no explicitly negative words
            - Mixed Sentiment: both positive and negative elements
            - Negation: negative meaning carried by structure not words
            - Other
    """
    print(f"\nError Analysis: First {n} Misclassified Examples:")
    errors = []
    for i, (true, pred) in enumerate(zip(y_test, y_pred)):
        if true != pred:
            errors.append({
                "index": i,
                "text": X_test_raw[i],
                "true": LABEL_MAP[true],
                "predicted": LABEL_MAP[pred]
            })

    for count, err in enumerate(errors[:n]):
        print(f"\nExample {count + 1}:")
        print(f"  Text     : {err['text']}")
        print(f"  True     : {err['true']}")
        print(f"  Predicted: {err['predicted']}")

    error_categories = [
        "Sarcasm",              # Example 1
        "Mixed Sentiment",      # Example 2
        "Sarcasm",              # Example 3
        "Implicit Negativity",  # Example 4
        "Implicit Negativity",  # Example 5
        "Negation",             # Example 6
        "Negation",             # Example 7
        "Implicit Negativity",  # Example 8
        "Mixed Sentiment",      # Example 9
        "Implicit Negativity",  # Example 10
    ]
 
    category_counts = {
        "Sarcasm": 0,
        "Implicit Negativity": 0,
        "Mixed Sentiment": 0,
        "Negation": 0,
        "Other": 0
    }

    for cat in error_categories:
        category_counts[cat] += 1
 
    # Print category breakdown
    print("\nError Category Breakdown:")
    for cat, count in category_counts.items():
        if count > 0:
            print(f"  {cat}: {count}")

    # Plot category breakdown
    plt.figure(figsize=(7, 4))
    plt.bar(category_counts.keys(), category_counts.values(), color='steelblue')
    plt.title("Naive Bayes — Error Categories (SST-2)")
    plt.ylabel("Count")
    plt.tight_layout()
    path = os.path.join(output_dir, "error_categories_nb.png")
    plt.savefig(path)
    plt.close()
    print(f"Error category chart saved to {path}")

    # Summary stats
    total_errors = len(errors)
    fn = sum(1 for t, p in zip(y_test, y_pred) if t == 1 and p == 0)
    fp = sum(1 for t, p in zip(y_test, y_pred) if t == 0 and p == 1)
    print(f"\nTotal misclassified : {total_errors}")
    print(f"  False Positives (Negative predicted as Positive): {fp}")
    print(f"  False Negatives (Positive predicted as Negative): {fn}")


def main():

    # loading, slicing, and splitting
    print("Loading the SST-2 dataset (GLUE benchmark)...")
    dataset = load_dataset("glue", "sst2")
 
    train_data = dataset['train'].select(range(5000))
    test_data = dataset['validation']
    
    X_train_raw = train_data['sentence']
    y_train = train_data['label']
    
    X_test_raw = test_data['sentence']
    y_test = test_data['label']

    print(" processing text and extracting features...")
    
    X_train_clean = preprocess_batch(X_train_raw)
    X_test_clean = preprocess_batch(X_test_raw)

    # using scikit library to convert raw text
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=None,       # keep all words including negations
        ngram_range=(1, 2),    # bigrams help capture negation and phrases
        sublinear_tf=True      # log scaling reduces dominance of frequent words
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)

    print("Training the Naive Bayes Baseline Model...")

    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    print("Evaluating the model...")
    y_pred = nb_model.predict(X_test_tfidf)
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nBaseline Accuracy: {accuracy * 100:.2f}%\n")
    
    print("Classification Report:")

    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
    
    # Confusion Matrix
    os.makedirs("results", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, output_dir="results")
 
    # Error Analysis
    error_analysis(X_test_raw, y_test, y_pred, output_dir="results", n=10)

if __name__ == "__main__":
    main()