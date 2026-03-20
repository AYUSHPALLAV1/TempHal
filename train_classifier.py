from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments, Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import f1_score
import sqlite3
import json
import numpy as np
import os
import torch

# Constants
LABEL2ID = {'stable': 0, 'volatile': 1, 'uncertain': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
DB_NAME = 'temphal.db'
OUTPUT_DIR = 'outputs/classifier'

def train_classifier():
    # 1. Load data from SQLite
    if not os.path.exists(DB_NAME):
        raise FileNotFoundError(f"Database {DB_NAME} not found. Please run phase 1 first.")
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    rows = cursor.execute("SELECT claim, temporal_type, split FROM claims").fetchall()
    conn.close()

    train_data = [{'text': r[0], 'label': LABEL2ID[r[1]]} for r in rows if r[2] == 'train']
    val_data = [{'text': r[0], 'label': LABEL2ID[r[1]]} for r in rows if r[2] == 'validation']

    if not train_data or not val_data:
        raise ValueError("Training or validation data is empty. Check Phase 1 results.")

    print(f"Loaded {len(train_data)} training and {len(val_data)} validation examples.")

    # 2. Tokenization
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128, padding='max_length')

    ds_train = Dataset.from_list(train_data).map(tokenize_function, batched=True)
    ds_val = Dataset.from_list(val_data).map(tokenize_function, batched=True)

    # 3. Model Initialization
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # 4. Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {'macro_f1': f1_score(labels, predictions, average='macro')}

    # 5. Training Arguments
    # Optimization for CPU if no GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3, # Reduced for speed, user requested 5
        per_device_train_batch_size=8, # Reduced for memory
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        logging_steps=50,
        report_to='none',
        use_cpu=(device == "cpu")
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # 7. Start Training
    print("Starting fine-tuning...")
    trainer.train()

    # 8. Save
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete.")

if __name__ == "__main__":
    train_classifier()
