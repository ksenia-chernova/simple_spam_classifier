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

# --- УНИВЕРСАЛЬНЫЙ БЛОК ВИЗУАЛИЗАЦИИ (после trainer.train()) ---

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np
import json
import os

os.makedirs('results', exist_ok=True)

# 1. Парсинг истории обучения
history = trainer.state.log_history

# Тренировочные потери (логгируются по шагам)
train_steps = [h['step'] for h in history if 'loss' in h and 'eval_loss' not in h]
train_loss = [h['loss'] for h in history if 'loss' in h and 'eval_loss' not in h]

# Метрики валидации (логгируются по эпохам)
eval_data = [h for h in history if 'eval_loss' in h]
eval_epochs = [h['epoch'] for h in eval_data]
eval_loss = [h['eval_loss'] for h in eval_data]
eval_f1 = [h.get('eval_f1') for h in eval_data]
eval_accuracy = [h.get('eval_accuracy') for h in eval_data]
eval_precision = [h.get('eval_precision') for h in eval_data]
eval_recall = [h.get('eval_recall') for h in eval_data]

print(f"📊 Train loss entries: {len(train_loss)}, Eval entries: {len(eval_data)}")

# 2. График тренировочных потерь (по шагам)
if train_steps and train_loss:
    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_loss, label='Train Loss', marker='.', markersize=2, alpha=0.7)
    plt.xlabel('Шаг обучения')
    plt.ylabel('Потери (Cross-Entropy)')
    plt.title('Динамика тренировочных потерь')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/train_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

# 3. График метрик валидации (по эпохам) - только если есть данные
if eval_epochs:
    plt.figure(figsize=(9, 5))
    
    # Словарь метрик для удобного перебора
    metrics = {
        'F1-Score': eval_f1,
        'Accuracy': eval_accuracy,
        'Precision': eval_precision,
        'Recall': eval_recall
    }
    
    plotted = False
    for name, values in metrics.items():
        # Фильтруем None-значения и проверяем длину
        valid_values = [v for v in values if v is not None]
        if len(valid_values) == len(eval_epochs) and valid_values:
            plt.plot(eval_epochs, valid_values, label=name, marker='o', linewidth=2)
            plotted = True
    
    if plotted:
        plt.xlabel('Эпоха')
        plt.ylabel('Значение')
        plt.title('Метрики на валидационной выборке')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(eval_epochs)  # Чёткие метки по эпохам
        plt.tight_layout()
        plt.savefig('results/val_metrics_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("⚠️ Нет валидных метрик для построения графика")

# 4. Предсказания на тесте
print("\n🔄 Генерация предсказаний...")
predictions = trainer.predict(tokenized_test)
pred_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = tokenized_test['label']
spam_probs = predictions.predictions[:, 1]  # Вероятности класса "Спам"

# 5. Матрица ошибок
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Не спам', 'Спам'],
            yticklabels=['Не спам', 'Спам'])
plt.xlabel('Предсказано')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок (тест)')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. ROC-кривая
fpr, tpr, _ = roc_curve(true_labels, spam_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Случайный классификатор')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Текстовый отчёт
print("\n" + "="*70)
print("📋 ОТЧЁТ О КЛАССИФИКАЦИИ")
print("="*70)
print(classification_report(true_labels, pred_labels,
                           target_names=['Не спам (0)', 'Спам (1)'],
                           digits=4))

# 8. Анализ ошибок
misclassified = []
for i, (pred, true, prob) in enumerate(zip(pred_labels, true_labels, spam_probs)):
    if pred != true:
        misclassified.append({
            'text': str(test_dataset[i]['text'])[:250] + '...',
            'true': int(true),
            'pred': int(pred),
            'prob': float(prob),
            'type': 'FP' if pred == 1 else 'FN'
        })

if misclassified:
    fp = sum(1 for e in misclassified if e['type'] == 'FP')
    fn = sum(1 for e in misclassified if e['type'] == 'FN')
    print(f"\n🔍 Ошибки: {len(misclassified)}/{len(true_labels)} ({100*len(misclassified)/len(true_labels):.2f}%)")
    print(f"   • Ложные срабатывания (FP): {fp}")
    print(f"   • Пропущенный спам (FN): {fn}")
    
    print("\n📝 Примеры ошибок:")
    for idx, item in enumerate(misclassified[:3], 1):
        lm = {0: 'Не спам', 1: 'Спам'}
        print(f"\n{idx}. [{item['type']}] \"{item['text']}\"")
        print(f"   Истинный: {lm[item['true']]} | Предсказано: {lm[item['pred']]} | Уверенность: {item['prob']:.2%}")

# 9. Сохранение метрик
final_metrics = {
    'accuracy': float(accuracy_score(true_labels, pred_labels)),
    'precision': float(precision_score(true_labels, pred_labels)),
    'recall': float(recall_score(true_labels, pred_labels)),
    'f1': float(f1_score(true_labels, pred_labels)),
    'roc_auc': float(roc_auc),
    'total_errors': len(misclassified),
    'false_positives': sum(1 for e in misclassified if e['type'] == 'FP'),
    'false_negatives': sum(1 for e in misclassified if e['type'] == 'FN')
}

with open('results/final_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(final_metrics, f, ensure_ascii=False, indent=2)

print(f"\n✅ Результаты сохранены в 'results/'")
print(f"✅ Итог: F1={final_metrics['f1']:.4f}, AUC={final_metrics['roc_auc']:.4f}")

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