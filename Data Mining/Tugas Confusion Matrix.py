import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Data asli
file_path = r'D:\Data\Home\Data Kuliah\S2\Data Mining\Dataset Tugas Data Mining Liner Regression.xlsx'
data = pd.read_excel(file_path)

# Data prediksi
status_kelulusan_prediksi = ['Lulus', 'Lulus', 'Lulus', 'Lulus',
                             'Lulus', 'Tidak Lulus', 'Lulus', 'Tidak Lulus', 'Lulus', 'Lulus']

# Confusion Matrix
conf_matrix = confusion_matrix(
    data['Status Kelulusan'], status_kelulusan_prediksi)

# Menampilkan Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[
            'Lulus', 'Tidak Lulus'], yticklabels=['Lulus', 'Tidak Lulus'])
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.title('Confusion Matrix')
plt.show()

# Menghitung akurasi
akurasi = accuracy_score(data['Status Kelulusan'], status_kelulusan_prediksi)
print(f"Akurasi: {akurasi*100:.2f}%")
