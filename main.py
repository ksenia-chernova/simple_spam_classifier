import torch
from datasets import load_dataset, concatenate_datasets, Value
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import numpy as np
from google.colab import files
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

ds = load_dataset("DmitryKRX/anti_spam_ru")['train']

ds = ds.train_test_split(test_size=0.2, seed=42)
train_dataset = ds['train']
test_dataset = ds['test']

print(f"Размер тренировки: {len(train_dataset)}, тест: {len(test_dataset)}")

model_name = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

MAX_LENGTH = 128

def tokenize_function(examples):
    texts = [str(t) if t is not None else "" for t in examples["text"]]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LENGTH)

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

tokenized_train = tokenized_train.rename_column("is_spam", "label")
tokenized_test = tokenized_test.rename_column("is_spam", "label")

tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions)
    }

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model = model.to(device)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32 if torch.cuda.is_available() else 8,
    per_device_eval_batch_size=32 if torch.cuda.is_available() else 8,
    num_train_epochs=2,
    weight_decay=0.02,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=100,
    report_to="none",
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()
print(results)

def predict_spam(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    label_map = {0: "Не спам", 1: "Спам"}
    return label_map[prediction], confidence

model.save_pretrained("./final_spam_model")
tokenizer.save_pretrained("./final_spam_model")

shutil.make_archive("final_spam_model", 'zip', "./final_spam_model")
files.download("final_spam_model.zip")