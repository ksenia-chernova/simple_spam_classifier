from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import shap
import numpy as np

model_path = "./final_spam_model_v3.0"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

def predict_spam(text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence = probs[0][1].item()
        prediction = "Спам" if confidence >= threshold else "Не спам"

    return prediction, confidence

def explain_with_shap(text):
    def predict_proba(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        elif not isinstance(texts, list):
            texts = [str(texts)]
        else:
            texts = [str(t) for t in texts]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return probs.cpu().numpy()

    explainer = shap.Explainer(
        predict_proba,
        tokenizer,
        output_names=["Не спам", "Спам"],
        algorithm="partition"
    )

    shap_values = explainer([text])
    shap.plots.text(shap_values)

    explanation = shap_values[0]
    spam_scores = explanation[:, "Спам"].values
    words = explanation.data[0]

    word_importance = sorted(
        zip(words, spam_scores),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    return word_importance, shap_values

def explain_with_shap_simple(text):
    def predict_proba(texts):
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        if not isinstance(texts, list):
            texts = [str(texts)]
        else:
            texts = [str(t) for t in texts]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return probs.cpu().numpy()

    masker = shap.maskers.Text(tokenizer, mask_token="[MASK]", collapse_mask_token=True)
    explainer = shap.Explainer(predict_proba, masker, output_names=["Не спам", "Спам"])

    shap_values = explainer([text])

    explanation = shap_values[0]
    spam_scores = explanation[:, "Спам"].values
    words = explanation.data[0]

    word_importance = []
    for word, score in zip(words, spam_scores):
        if word and not word.startswith('[') and len(word.strip()) > 1:
            word_importance.append((word, score))

    word_importance.sort(key=lambda x: abs(x[1]), reverse=True)

    return word_importance[:5]

if __name__ == "__main__":
    print("🤖 Антиспам-классификатор с SHAP-объяснениями")
    print("Команды: 'explain текст' - подробное объяснение, 'simple текст' - быстрое объяснение, 'exit' - выход\n")

    while True:
        user_input = input("\n📱 Ваш текст: ")

        if user_input.lower() in ['exit', 'quit', 'выход']:
            break
        if not user_input.strip():
            continue

        if user_input.lower().startswith('explain '):
            text = user_input[7:].strip()
            label, conf = predict_spam(text)
            status = "🚫 СПАМ" if label == "Спам" else "✅ НЕ СПАМ"
            print(f"\n📊 Результат: {status} | Уверенность: {conf:.2%}")

            try:
                print("\n🔍 Анализ SHAP (ключевые слова):")
                important_words, _ = explain_with_shap(text)

                for word, score in important_words:
                    if score > 0:
                        print(f"   🔴 '{word}' → +{score:.3f} (признак спама)")
                    else:
                        print(f"   🟢 '{word}' → {score:.3f} (признак НЕ спама)")
            except Exception as e:
                print(f"❌ Ошибка при объяснении: {e}")
                print("💡 Попробуйте команду 'simple' для быстрого объяснения")

        elif user_input.lower().startswith('simple '):
            text = user_input[7:].strip()
            label, conf = predict_spam(text)
            status = "🚫 СПАМ" if label == "Спам" else "✅ НЕ СПАМ"
            print(f"\n📊 Результат: {status} | Уверенность: {conf:.2%}")

            print("\n🔍 Быстрый анализ (ключевые слова):")
            important_words = explain_with_shap_simple(text)

            for word, score in important_words:
                if score > 0:
                    print(f"   🔴 '{word}' → +{score:.3f} (признак спама)")
                else:
                    print(f"   🟢 '{word}' → {score:.3f} (признак НЕ спама)")

        else:
            label, conf = predict_spam(user_input)
            status = "🚫 СПАМ" if label == "Спам" else "✅ НЕ СПАМ"
            print(f"📊 Результат: {status} | Уверенность: {conf:.2%}")
            print("💡 Команды с объяснением: 'explain текст' или 'simple текст'")