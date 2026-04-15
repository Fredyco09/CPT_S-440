import os
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_NAME = "distilbert-base-uncased"
SAVE_PATH = "./results"
LABEL_MAP = {0: "Negative", 1: "Positive"}


def tokenize(example, tokenizer):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=64
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def plot_confusion_matrix(cm, output_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title("DistilBERT — Confusion Matrix (SST-2)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix_nn.png")
    plt.savefig(path)
    plt.close()
    print(f"Confusion matrix saved to {path}")
 
 
def error_analysis(X_test_raw, y_test, y_pred, output_dir, n=10):
    """
    Prints misclassified examples and plots error category breakdown.
    Uses the same categories as nb_model.py for direct comparison:
        - Sarcasm
        - Implicit Negativity
        - Mixed Sentiment
        - Negation
        - Other
    """
    print(f"\n--- Error Analysis: First {n} Misclassified Examples ---")
    errors = []
    for i, (true, pred) in enumerate(zip(y_test, y_pred)):
        if true != pred:
            errors.append({
                "index": i,
                "text": X_test_raw[i],
                "true": LABEL_MAP[true],
                "predicted": LABEL_MAP[pred]
            })
 
    # Print first n errors
    for count, err in enumerate(errors[:n]):
        print(f"\nExample {count + 1}:")
        print(f"  Text     : {err['text']}")
        print(f"  True     : {err['true']}")
        print(f"  Predicted: {err['predicted']}")
 

    error_categories = [
        "Mixed Sentiment",      # Example 1
        "Implicit Negativity",  # Example 2
        "Implicit Negativity",  # Example 3
        "Implicit Negativity",  # Example 4
        "Mixed Sentiment",      # Example 5
        "Implicit Negativity",  # Example 6
        "Negation",             # Example 7
        "Mixed Sentiment",      # Example 8
        "Negation",             # Example 9
        "Sarcasm",              # Example 10
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
    plt.bar(category_counts.keys(), category_counts.values(), color='darkorange')
    plt.title("DistilBERT — Error Categories (SST-2)")
    plt.ylabel("Count")
    plt.tight_layout()
    path = os.path.join(output_dir, "error_categories_nn.png")
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
    print("loading dataset...")
    dataset = load_dataset("glue", "sst2")

    train_data = dataset["train"].select(range(10000))
    test_data = dataset["validation"]

    X_test_raw = test_data["sentence"]

    print("loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    print("tokenizing...")
    train_data = train_data.map(lambda x: tokenize(x, tokenizer), batched=True)
    test_data = test_data.map(lambda x: tokenize(x, tokenizer), batched=True)

    train_data = train_data.remove_columns(["sentence", "idx"])
    test_data = test_data.remove_columns(["sentence", "idx"])

    train_data = train_data.rename_column("label", "labels")
    test_data = test_data.rename_column("label", "labels")

    train_data.set_format("torch")
    test_data.set_format("torch")

    print("training...")

    args = TrainingArguments(
        output_dir=SAVE_PATH,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        save_total_limit=1,              # Only keep the best checkpoint; delete old ones
        dataloader_pin_memory=False,     # Disable for Mac/MPS compatibility
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Evaluate
    print("Evaluating...")
    results = trainer.evaluate()
    print(f"\nAccuracy: {results['eval_accuracy'] * 100:.2f}%\n")
 
    # Get predictions for full classification report + confusion matrix
    preds_output = trainer.predict(test_data)
    y_pred = np.argmax(preds_output.predictions, axis=1)
    y_test = preds_output.label_ids
 
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
 
    # Confusion Matrix
    os.makedirs("results", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, output_dir="results")
 
    # Error Analysis
    error_analysis(X_test_raw, y_test, y_pred, output_dir="results", n=10)
 
    # Save model
    print("Saving model...")
    trainer.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("Done.")


if __name__ == "__main__":
    main()