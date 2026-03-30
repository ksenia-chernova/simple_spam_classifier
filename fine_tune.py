import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

old_ds = load_dataset("DmitryKRX/anti_spam_ru")['train'].select(range(5000))
new_ds1 = load_dataset("ruSpamModels/russian-spam-detection")
new_ds2 = load_dataset("benzlokzik/russian-spam-fork")

ds = concatenate_datasets([old_ds, new_ds1, new_ds2])
print(len(ds))