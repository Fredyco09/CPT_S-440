import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("1. Loading the IMDB dataset...")
   
    dataset = load_dataset("imdb")
 
    train_data = dataset['train'][:5000]
    test_data = dataset['test'][:1000]
    
    X_train_raw = train_data['text']
    y_train = train_data['label']
    
    X_test_raw = test_data['text']
    y_test = test_data['label']

    print("2. Preprocessing text and extracting features (TF-IDF)...")
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    

    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)

    print("3. Training the Naive Bayes Baseline Model...")

    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    print("4. Evaluating the model...")
    y_pred = nb_model.predict(X_test_tfidf)
    
    # Calculate basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nBaseline Accuracy: {accuracy * 100:.2f}%\n")
    
    print("Classification Report:")

    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
    
    


if __name__ == "__main__":
    main()