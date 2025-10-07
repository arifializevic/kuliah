import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
df = pd.read_csv(url, sep=";")
data = df.head(100)  # Ambil 100 data pertama

# Analisis Pemusatan Data (mean, median, modus)
mean_values = data.mean()
median_values = data.median()
mode_values = data.mode().iloc[0]

print("=== Pemusatan Data ===")
print("Mean:\n", mean_values)
print("\nMedian:\n", median_values)
print("\nModus:\n", mode_values)

# Analisis Penyebaran Data (range, variance, standard deviation)
range_values = data.max() - data.min()
variance_values = data.var()
std_dev_values = data.std()

print("\n=== Penyebaran Data ===")
print("Range:\n", range_values)
print("\nVariance:\n", variance_values)
print("\nStandard Deviation:\n", std_dev_values)

# Pendugaan parameter rata-rata (confidence interval 99%)
conf_level = 0.99
sample_mean = data['alcohol'].mean()
sample_std = data['alcohol'].std()
n = len(data)

z_score = stats.norm.ppf(1 - (1 - conf_level)/2)
margin_error = z_score * (sample_std / np.sqrt(n))
lower_bound = sample_mean - margin_error
upper_bound = sample_mean + margin_error

print("\n=== Pendugaan Parameter Rata-rata Alkohol (CI 99%) ===")
print(f"Mean: {sample_mean:.4f}")
print(f"Confidence Interval: ({lower_bound:.4f}, {upper_bound:.4f})")

# Visualisasi Distribusi Data
plt.figure(figsize=(12, 6))

# Histogram untuk kolom 'alcohol'
plt.subplot(1, 2, 1)
sns.histplot(data['alcohol'], kde=True, bins=20, color='blue')
plt.title('Distribusi Alkohol (Histogram & KDE)')
plt.xlabel('Alkohol (%)')
plt.ylabel('Frekuensi')

# Boxplot untuk melihat penyebaran data
plt.subplot(1, 2, 2)
sns.boxplot(x=data['alcohol'], color='lightblue')
plt.title('Boxplot Alkohol')
plt.xlabel('Alkohol (%)')

plt.tight_layout()
plt.show()