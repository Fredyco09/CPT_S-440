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
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train_clean)
    X_test_tfidf = vectorizer.transform(X_test_clean)

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
    
    # Confusion
    os.makedirs("results", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title("Naive Bayes — Confusion Matrix (SST-2)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_nb.png")
    plt.close()
    print("Confusion matrix saved to results/confusion_matrix_nb.png")
 
    # Error analysis
    print("\n--- Error Analysis: First 10 Misclassified Examples ---")
    label_map = {0: "Negative", 1: "Positive"}
    count = 0
    for i, (true, pred) in enumerate(zip(y_test, y_pred)):
        if true != pred:
            print(f"\nExample {count + 1}:")
            print(f"  Text     : {X_test_raw[i]}")
            print(f"  True     : {label_map[true]}")
            print(f"  Predicted: {label_map[pred]}")
            count += 1
            if count >= 10:
                break

if __name__ == "__main__":
    main()