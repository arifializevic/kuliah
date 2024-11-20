# Import library
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

# Unduh resource NLTK (jika belum diunduh)
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# === 1. Memuat Data ===
# Ganti dengan file dataset Anda
data = pd.read_csv('dataset\data_sentimen.csv')  # Kolom: 'text', 'label'

print("Data Awal:")
print(data.head())

# === 2. Pra-pemrosesan Teks ===


def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Hapus karakter non-alfabet
    text = text.lower()  # Ubah ke huruf kecil
    text = word_tokenize(text)  # Tokenisasi
    text = [word for word in text if word not in stopwords.words(
        'english')]  # Hapus stopwords
    return ' '.join(text)


# Terapkan pra-pemrosesan
data['cleaned_text'] = data['text'].apply(preprocess_text)

print("\nData Setelah Pra-pemrosesan:")
print(data.head())

# === 3. Ekstraksi Fitur dengan TF-IDF ===
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned_text']).toarray()
y = data['label']  # Kolom label (misalnya: 'positif', 'negatif')

# === 4. Membagi Data untuk Training dan Testing ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# === 5. Melatih Model Machine Learning ===
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediksi dan Evaluasi
y_pred = model.predict(X_test)
print("\nHasil Evaluasi Model:")
print(classification_report(y_test, y_pred))
print(f"Akurasi: {accuracy_score(y_test, y_pred):.2f}")

# === 6. Visualisasi Data Sentimen ===
plt.figure(figsize=(8, 6))
sns.countplot(x=y)
plt.title('Distribusi Sentimen')
plt.xlabel('Label Sentimen')
plt.ylabel('Jumlah')
plt.show()

# === 7. Analisis Sentimen dengan Model Pre-trained (Opsional) ===
# Menggunakan Model BERT melalui Transformers
sentiment_pipeline = pipeline("sentiment-analysis")
sample_texts = ["I love this product!", "This is the worst experience ever."]
bert_results = sentiment_pipeline(sample_texts)

print("\nAnalisis Sentimen Menggunakan BERT:")
for text, result in zip(sample_texts, bert_results):
    print(f"Teks: {text}")
    print(f"Hasil: {result}\n")
