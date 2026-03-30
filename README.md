# Simple Spam Classifier - Простой классификатор спама в SMS/Email

Простой классификатор спама на русском языке, построенный на основе трансформерной модели BERT. Пет-проект для изучения NLP и машинного обучения.
Проект представляет собой модель бинарной классификации текстовых сообщений на спам и не спам. Модель обучена на русскоязычном датасете и использует предобученную модель rubert-tiny2 для достижения высокого качества при минимальных требованиях к вычислительным ресурсам.

### Установка
1. Склонируйте данный репозиторий
```bash
git clone git@github.com:ksenia-chernova/simple_spam_classifier.git
```
2. Установите зависимости
```bash
pip install datasets transformers scikit-learn torch accelerate evaluate
```
3. Запустите файл для обучения модели
```bash
python main.py
```
4. Запустите файл для работы с обученной моделью
```bash
python predict.py
```

### Пример использования
<img width="1090" height="229" alt="image" src="https://github.com/user-attachments/assets/6cd3df9b-af8b-4dbd-89fb-2489c73421ed" />

### Ссылки
1. [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
2. [Datasets Library](https://huggingface.co/docs/datasets/index)
3. [rubert-tiny2 модель](https://huggingface.co/cointegrated/rubert-tiny2?spm=a2ty_o01.29997173.0.0.74fe5171r5Y7r5)
4. [Датасет anti_spam_ru](https://huggingface.co/datasets/DmitryKRX/anti_spam_ru?spm=a2ty_o01.29997173.0.0.74fe5171r5Y7r5)
