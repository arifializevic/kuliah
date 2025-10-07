from scipy.stats import friedmanchisquare

# Data waktu eksekusi (dalam detik) untuk 15 dataset
algo_a = [2.1, 2.3, 2.0, 2.4, 2.2, 2.1, 2.3, 2.2, 2.1, 2.4, 2.2, 2.3, 2.1, 2.2, 2.3]
algo_b = [3.2, 3.5, 3.1, 3.4, 3.3, 3.0, 3.4, 3.3, 3.2, 3.5, 3.1, 3.4, 3.3, 3.0, 3.2]
algo_c = [1.8, 1.9, 1.7, 2.0, 1.8, 1.9, 1.8, 1.9, 1.7, 2.0, 1.8, 1.9, 1.8, 1.9, 1.7]

# Uji Friedman
stat, p = friedmanchisquare(algo_a, algo_b, algo_c)
print(f"Statistic: {stat:.4f}, p-value: {p:.6f}")
