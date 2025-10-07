import numpy as np
from scipy.stats import ttest_1samp

# Data
sample_mean = 11.39  # Rata-rata sampel
population_mean = 13.17  # Rata-rata populasi yang dihipotesiskan
std_dev = 2.09  # Standar deviasi sampel
n = 36  # Ukuran sampel

# Data sampel (simulasi)
sample_data = np.random.normal(sample_mean, std_dev, n)

# Uji hipotesis menggunakan ttest_1samp
t_stat, p_value = ttest_1samp(sample_data, population_mean)

# Output hasil
print(f"Nilai t-statistic: {t_stat}")
print(f"Nilai p-value: {p_value}")

# Keputusan berdasarkan p-value
alpha = 0.05
if p_value < alpha:
    print("Kesimpulan: Tolak H0. Rata-rata hasil investasi berbeda signifikan dari 10,23%.")
else:
    print("Kesimpulan: Gagal menolak H0. Tidak ada bukti bahwa rata-rata hasil investasi berbeda dari 10,23%.")
