import pandas as pd
import numpy as np

# Membuat dataset sintetis
np.random.seed(42)
data = {
    'Harga_Iklan': np.random.normal(500, 100, 30),  # Variabel bebas 1 (dalam ribuan)
    'Jumlah_Outlet': np.random.randint(10, 50, 30),  # Variabel bebas 2
    'Penjualan': 80 + 0.5 * np.random.normal(500, 100, 30) + 
                  3 * np.random.randint(10, 50, 30) + 
                  np.random.normal(0, 20, 30)  # Variabel terikat
}

df = pd.DataFrame(data)
print(df.head())