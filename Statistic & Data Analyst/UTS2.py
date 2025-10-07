import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# (a) Hipotesis
# H0: μ >= 2
# H1: μ < 2

# (b) Statistik uji
mu_0 = 2
x_bar = 1.8
std_dev = 0.5
n = 200
z = (x_bar - mu_0) / (std_dev / np.sqrt(n))

# (c) Nilai kritis dan p-value
z_critical = stats.norm.ppf(0.05)  # Karena H1: μ < 2
p_value = stats.norm.cdf(z)

print("\n=== Uji Hipotesis Waktu Respon Server ===")
print(f"Statistik Z: {z:.4f}")
print(f"Z Kritis (α=0.05): {z_critical:.4f}")
print(f"P-Value: {p_value:.4f}")

# (d) Kesimpulan
if z < z_critical:
    print("Kesimpulan: Tolak H0. Waktu respon server kurang dari 2 detik.")
else:
    print("Kesimpulan: Gagal tolak H0. Tidak cukup bukti bahwa waktu respon kurang dari 2 detik.")

# Visualisasi Distribusi Normal
x = np.linspace(mu_0 - 4 * (std_dev / np.sqrt(n)),
                mu_0 + 4 * (std_dev / np.sqrt(n)), 1000)
y = stats.norm.pdf(x, loc=mu_0, scale=std_dev / np.sqrt(n))

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Distribusi Normal', color='blue')

# Area kritis (daerah penolakan H0)
x_critical = np.linspace(-np.inf, z_critical, 1000)
y_critical = stats.norm.pdf(x_critical, loc=mu_0, scale=std_dev / np.sqrt(n))
plt.fill_between(x_critical, y_critical, color='red',
                 alpha=0.3, label='Area Kritis')

# Titik statistik uji
plt.axvline(x=x_bar, color='green', linestyle='--',
            label=f'Statistik Uji (Z={z:.2f})')

plt.title('Uji Hipotesis dengan Distribusi Normal')
plt.xlabel('Waktu Respon (detik)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid()
plt.show()
