import asyncio
import os
import re
import django
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from asgiref.sync import sync_to_async
from sklearn.svm import SVC

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'MetinProcessing.settings')
django.setup()
from MetinApp.models import ProcessedTexts

os.makedirs("ML_Model", exist_ok=True)


async def load_data_from_db(start=0, limit=16000):
    """
    Veritabanından veri yükler. Belirli bir başlangıç noktasından (start)
    belirli bir limit (limit) kadar veri çeker.
    Yalnızca `textType` değeri 0 ile 4 arasında olan verileri getirir.
    """
    try:
        data = await sync_to_async(
            lambda: list(
                ProcessedTexts.objects.filter(textType__gt=0, textType__lte=4)
                .order_by('id')[start:start + limit]
                .values('processed_text', 'textType')
            )
        )()
        df = pd.DataFrame(data)
        if df.empty:
            print("Veri boş!")
        return df
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return pd.DataFrame()


def clean_data(text):
    """
    Verileri, modelin işleyebileceği formata getirmek için gerekli
    veri temizleme işlemlerini gerçekleştirir:
    - Noktalama işaretlerini kaldırır.
    - Küçük harfe çevirir.
    - Durdurma kelimelerini (stopwords) çıkarır.
    - Kelimelerin kök halini (lemmatization) alır.
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[#$@{}\[\]\/\\)<>(|!\'^+%&/½=*&€~¨´æ£éß]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()
    cleaned_text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return cleaned_text


def train_and_save_model():
    """
    Veritabanından veri yükler, model eğitimi gerçekleştirir ve eğitilen modeli
    disk üzerine kaydeder. Model `RandomForestClassifier` kullanılarak eğitilir.
    - Veriler TfidfVectorizer ile sayısallaştırılır.
    - Eğitim ve test verileri oluşturulur.
    - Model eğitilir ve doğruluk skorları hesaplanır.
    """
    data = asyncio.run(load_data_from_db())
    if data.empty:
        print("Yetersiz veri ile model eğitimi yapılamaz.")
        return

    X = data['processed_text']
    y = data['textType']

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Doğruluk:", accuracy)
    print("Sınıflandırma Raporu:\n", report)

    model_path = os.path.join("ML_Model", "trained_model.joblib")
    vectorizer_path = os.path.join("ML_Model", "vectorizer.joblib")
    dump(model, model_path)
    dump(vectorizer, vectorizer_path)
    print(f"Model ve vektörizer kaydedildi: {model_path}, {vectorizer_path}")


def predict_class(text):
    """
    Girdi metnini sınıflandırır ve tahmin edilen sınıfı döndürür.
    - Kaydedilen model ve vektörizer kullanılarak tahmin yapılır.
    - Model, metni sayısallaştırır ve sınıfı döndürür.
    """
    model_path = os.path.join("ML_Model", "trained_model.joblib")
    vectorizer_path = os.path.join("ML_Model", "vectorizer.joblib")

    try:
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            model = load(model_path)
            vectorizer = load(vectorizer_path)
        else:
            raise FileNotFoundError("Model veya vektörizer bulunamadı.")

        cleaned_text = clean_data(text)  # Metni temizle
        text_vector = vectorizer.transform([cleaned_text])
        predicted_class_int = model.predict(text_vector)[0]

        int_to_class = {1: "World", 2: "Sports", 3: "Business", 4: "Science/Technology",
                        5: "Entertainment", 6: "Politics", 7: "Medical"}
        predicted_class_name = int_to_class.get(predicted_class_int, "Bilinmeyen Sınıf")
        return predicted_class_name
    except Exception as e:
        print(f"Tahmin hatası: {e}")
        return "Tahmin Hatası"


def compare_models():
    """
    Farklı makine öğrenimi modellerinin performansını karşılaştırır.
    - `Naive Bayes`, `Lojistik Regresyon`, `Rastgele Orman` ve `SVM` kullanılır.
    - Her modelin doğruluk skorları hesaplanır ve karşılaştırılır.
    """
    data = asyncio.run(load_data_from_db())
    if data.empty:
        print("Yetersiz veri ile model karşılaştırması yapılamaz.")
        return

    X = data['processed_text']
    y = data['textType']

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Lojistik Regresyon": LogisticRegression(max_iter=2000),
        "Rastgele Orman": RandomForestClassifier(),
        "SVM": SVC(kernel='linear', max_iter=2000)
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model: {model_name} - Doğruluk: {accuracy:.4f}")

# def transfer_learn(new_data, new_labels, model_path="ML_Model/trained_model.joblib", vectorizer_path="ML_Model/vectorizer.joblib"):
#     """
#     Eğitilmiş modele transfer öğrenme yapar:
#     - Yeni veri ve etiketlerle mevcut modeli yeniden eğitir.
#     - Güncellenmiş modeli kaydeder.
    
#     Args:
#         new_data (list): Yeni eğitim metinleri.
#         new_labels (list): Yeni metinlerin etiketleri.
#         model_path (str): Kaydedilen modelin yolu.
#         vectorizer_path (str): Kaydedilen vektörizerin yolu.
#     """
#     try:
#         if os.path.exists(model_path) and os.path.exists(vectorizer_path):
#             model = load(model_path)
#             vectorizer = load(vectorizer_path)
#         else:
#             raise FileNotFoundError("Model veya vektörizer bulunamadı.")

#         cleaned_data = [clean_data(text) for text in new_data]
#         new_X = vectorizer.transform(cleaned_data)

#         if hasattr(model, 'partial_fit'):
#             all_classes = list(set(new_labels))
#             model.partial_fit(new_X, new_labels, classes=all_classes)
#         else:
#             print("Model incremental learning desteklemiyor. Full fit için eski veriler gerekli.")
#             return

#         dump(model, model_path)
#         print(f"Transfer öğrenme tamamlandı ve model güncellendi: {model_path}")

#     except Exception as e:
#         print(f"Transfer öğrenme hatası: {e}")

