import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

# Dataset
df = pd.read_csv('dataset_tugas11.csv')

# Menentukan variabel dependen dan independen
X = df[['Kualitas_Produk', 'Lama_Pengiriman','Pembayaran', 'Promo', 'Voucher', 'Kesesuaian_Produk']]
Y = df['Kepuasan_Pelanggan']

# Menambahkan konstanta untuk model regresi
X = sm.add_constant(X)

# Menjalankan model regresi linier
model = sm.OLS(Y, X).fit()

# Menampilkan ringkasan hasil regresi
print("Ringkasan Model Regresi Linier:")
print(model.summary())

# Uji Normalitas (Shapiro-Wilk)
residuals = model.resid
shapiro_test = stats.shapiro(residuals)
print("\nUji Normalitas Shapiro-Wilk:")
print(f"Statistik: {shapiro_test[0]}, p-value: {shapiro_test[1]}")

# Uji Multikolinearitas (VIF)
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print("\nUji Multikolinearitas (VIF):")
print(vif_data)

# Uji Heteroskedastisitas (Breusch-Pagan)
bp_test = het_breuschpagan(residuals, X)
print("\nUji Heteroskedastisitas (Breusch-Pagan):")
print(f"Statistik: {bp_test[0]}, p-value: {bp_test[1]}")

# Uji Autokorelasi (Durbin-Watson)
dw_test = durbin_watson(residuals)
print("\nUji Autokorelasi (Durbin-Watson):")
print(f"Statistik Durbin-Watson: {dw_test}")
