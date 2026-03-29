from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./final_spam_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

def predict_spam(text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence = probs[0][1].item()
        prediction = "Спам" if confidence >= threshold else "Не спам"
    
    return prediction, confidence

if __name__ == "__main__":    
    while True:
        user_input = input("\nВаш текст: ")
        if user_input.lower() in ['exit', 'quit', 'выход']:
            break
        if not user_input.strip():
            continue
            
        label, conf = predict_spam(user_input)
        status = "🚫 СПАМ" if label == "Спам" else "✅ НЕ СПАМ"
        print(f"Результат: {status} | Уверенность: {conf:.2%}")