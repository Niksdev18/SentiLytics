# src/models/transformer.py
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib
from collections import Counter
from torch.nn import CrossEntropyLoss

# -----------------------------
# Ensure output directories exist
# -----------------------------
os.makedirs("models/transformer_model", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -----------------------------
# Load preprocessed datasets
# -----------------------------
train_df = pd.read_csv("data/processed/train.csv")
val_df   = pd.read_csv("data/processed/val.csv")
test_df  = pd.read_csv("data/processed/test.csv")

train_df = train_df.dropna(subset=['text','label'])
val_df   = val_df.dropna(subset=['text','label'])
test_df  = test_df.dropna(subset=['text','label'])

X_train, y_train = train_df['text'].fillna(""), train_df['label']
X_val, y_val     = val_df['text'].fillna(""), val_df['label']
X_test, y_test   = test_df['text'].fillna(""), test_df['label']

print("Train class distribution:", Counter(y_train))

# -----------------------------
# Encode labels
# -----------------------------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)
y_test_enc  = le.transform(y_test)

# Save LabelEncoder
joblib.dump(le, "models/transformer_model/label_encoder.pkl")

# -----------------------------
# Dataset class
# -----------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# -----------------------------
# Load FinBERT tokenizer & model
# -----------------------------
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(le.classes_)
)

# -----------------------------
# Prepare datasets
# -----------------------------
train_dataset = NewsDataset(X_train.tolist(), y_train_enc, tokenizer)
val_dataset   = NewsDataset(X_val.tolist(), y_val_enc, tokenizer)
test_dataset  = NewsDataset(X_test.tolist(), y_test_enc, tokenizer)

# -----------------------------
# Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="models/transformer_model/",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)
# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# -----------------------------
# Save model & tokenizer
# -----------------------------
trainer.save_model("models/transformer_model/best_model")
tokenizer.save_pretrained("models/transformer_model/best_model")

# -----------------------------
# Evaluate on test set
# -----------------------------
preds_output = trainer.predict(test_dataset)
y_pred = preds_output.predictions.argmax(-1)

accuracy = accuracy_score(y_test_enc, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test_enc, y_pred, average='weighted')

print(f"Transformer Test Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# -----------------------------
# Append metrics to model_metrics.csv
# -----------------------------
metrics_df = pd.DataFrame([{
    "Model": "Transformer_FinBERT",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1": f1
}])

metrics_file = "models/model_metrics.csv"
if os.path.exists(metrics_file):
    existing_metrics = pd.read_csv(metrics_file)
    metrics_df = pd.concat([existing_metrics, metrics_df], ignore_index=True)

metrics_df.to_csv(metrics_file, index=False)
print(f"Transformer metrics appended to {metrics_file}")
