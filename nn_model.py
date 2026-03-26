import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


MODEL_NAME = "distilbert-base-uncased"
SAVE_PATH = "./model_output"


def tokenize(example, tokenizer):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


def main():
    print("loading dataset...")
    dataset = load_dataset("glue", "sst2")

    train_data = dataset["train"].select(range(5000))
    test_data = dataset["validation"]

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
        per_device_train_batch_size=16,
        num_train_epochs=2,
        logging_steps=50,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=test_data,
    )

    trainer.train()

    print("saving model...")
    trainer.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    print("done")


if __name__ == "__main__":
    main()