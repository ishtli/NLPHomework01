from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 64
NUM_LABELS = 8

root = Path(__file__).resolve().parent
data_dir = root / "data" / "NYTprocessed"
output_dir = root / "outputs" / "task03_bert_01"


if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is unavailable. Check the PyTorch installation.")

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA version:", torch.version.cuda)


def read_split(name):
    frame = pd.read_csv(
        data_dir / f"{name}.csv",
        usecols=["text", "label"],
    )

    frame = frame.rename(columns={"label": "labels"})
    return Dataset.from_pandas(frame, preserve_index=False)


train_dataset = read_split("train")
val_dataset = read_split("val")
test_dataset = read_split("test")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


train_dataset = train_dataset.map(
    tokenize_batch,
    batched=True,
    remove_columns=["text"],
)
val_dataset = val_dataset.map(
    tokenize_batch,
    batched=True,
    remove_columns=["text"],
)
test_dataset = test_dataset.map(
    tokenize_batch,
    batched=True,
    remove_columns=["text"],
)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
)


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)
    true_labels = eval_pred.label_ids

    return {
        "accuracy": accuracy_score(true_labels, predictions),
        "macro_f1": f1_score(
            true_labels,
            predictions,
            average="macro",
        ),
    }


training_args = TrainingArguments(
    output_dir=str(output_dir),
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=200,
    report_to="none",
    seed=42,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

print("Best checkpoint:", trainer.state.best_model_checkpoint)

test_result = trainer.evaluate(
    test_dataset,
    metric_key_prefix="test",
)

print(f"Test Accuracy: {test_result['test_accuracy']:.6f}")
print(f"Test Macro-F1: {test_result['test_macro_f1']:.6f}")