from scipy.stats import kruskal

# Data waktu eksekusi
algo_a = [2.1, 2.3, 2.0, 2.4, 2.2, 2.1, 2.3, 2.2, 2.1, 2.4, 2.2, 2.3, 2.1, 2.2, 2.3]
algo_b = [3.2, 3.5, 3.1, 3.4, 3.3, 3.0, 3.4, 3.3, 3.2, 3.5, 3.1, 3.4, 3.3, 3.0, 3.2]
algo_c = [1.8, 1.9, 1.7, 2.0, 1.8, 1.9, 1.8, 1.9, 1.7, 2.0, 1.8, 1.9, 1.8, 1.9, 1.7]

stat, p = kruskal(algo_a, algo_b, algo_c)
print(f"Statistic: {stat:.4f}, p-value: {p:.6f}")