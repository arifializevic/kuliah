import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Diketahui
mu = 200  # Rata-rata waktu respons (milidetik)
sigma = 50  # Standar deviasi waktu respons (milidetik)

# (a) P(X < 250)
prob1 = stats.norm.cdf(250, loc=mu, scale=sigma)

# (b) P(200 < X < 270)
prob2 = stats.norm.cdf(270, loc=mu, scale=sigma) - stats.norm.cdf(200, loc=mu, scale=sigma)

print("\n=== Distribusi Normal Waktu Respon ===")
print(f"Probabilitas waktu respon < 250 ms: {prob1:.4f}")
print(f"Probabilitas waktu respon antara 200 dan 270 ms: {prob2:.4f}")

# Visualisasi Distribusi Normal
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
y = stats.norm.pdf(x, loc=mu, scale=sigma)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Distribusi Normal', color='blue')

# Area P(X < 250)
x1 = np.linspace(mu - 4 * sigma, 250, 1000)
y1 = stats.norm.pdf(x1, loc=mu, scale=sigma)
plt.fill_between(x1, y1, color='green', alpha=0.3, label=f'P(X < 250) = {prob1:.4f}')

# Area P(200 < X < 270)
x2 = np.linspace(200, 270, 1000)
y2 = stats.norm.pdf(x2, loc=mu, scale=sigma)
plt.fill_between(x2, y2, color='orange', alpha=0.3, label=f'P(200 < X < 270) = {prob2:.4f}')

plt.title('Distribusi Normal Waktu Respon Server')
plt.xlabel('Waktu Respon (milidetik)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid()
plt.show()