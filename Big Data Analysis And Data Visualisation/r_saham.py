# ===============================================
# 1. INSTALASI LIBRARY (JIKA BELUM)
# ===============================================
# Jalankan perintah ini di terminal atau command prompt Anda:
# pip install yfinance pandas statsmodels pmdarima prophet scikit-learn matplotlib seaborn

# ===============================================
# 2. IMPORT LIBRARY
# ===============================================
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# ===============================================
# 3. AMBIL DATA SAHAM MYOR.JK
# ===============================================
ticker = "MYOR.JK"
start_date = "2018-01-01"
end_date = "2025-05-05"

data = yf.download(ticker, start=start_date, end=end_date)

# ===============================================
# 4. BERSIHKAN DATA DAN AMBIL HARGA PENUTUPAN
# ===============================================
data.dropna(inplace=True)
close_data = data['Close']

print("===== Informasi Data Harga Penutupan =====")
print(f"Jumlah data: {len(close_data)}")
print("\n5 data pertama:")
print(close_data.head())
print("\n5 data terakhir:")
print(close_data.tail())
print("\nRingkasan Statistik:")
print(close_data.describe())


#################################################################
##                                                             ##
##                    ANALISIS DENGAN ARIMA                    ##
##                                                             ##
#################################################################

print("\n\n" + "="*50)
print(" " * 15 + "ANALISIS DENGAN ARIMA")
print("="*50 + "\n")


# ===============================
# 5. UJI STASIONERITAS (ADF TEST)
# ===============================
# H0: Data tidak stasioner (memiliki unit root)
# H1: Data stasioner
# -------------------------------
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    if result[1] <= 0.05:
        print("--> Kesimpulan: Data stasioner (tolak H0)")
    else:
        print("--> Kesimpulan: Data tidak stasioner (gagal tolak H0)")

print("===== Uji Stasioneritas Data Asli =====")
adf_test(close_data)

# Plot data asli
plt.figure(figsize=(12, 6))
plt.plot(close_data, label='Harga Penutupan Asli', color='blue')
plt.title('Harga Penutupan Saham MYOR.JK')
plt.xlabel('Tanggal')
plt.ylabel('Harga')
plt.legend()
plt.show()

# ===============================
# 6. TRANSFORMASI DIFFERENCING (JIKA PERLU)
# ===============================
# Karena data tidak stasioner, kita lakukan differencing
diff_close = close_data.diff().dropna()

print("\n===== Uji Stasioneritas Setelah Differencing (d=1) =====")
adf_test(diff_close)

# Plot data setelah differencing
plt.figure(figsize=(12, 6))
plt.plot(diff_close, label='Harga Penutupan (Differenced)', color='red')
plt.title('Harga Penutupan MYOR.JK Setelah Differencing')
plt.xlabel('Tanggal')
plt.ylabel('Perbedaan Harga')
plt.legend()
plt.show()

# ===============================
# 7. IDENTIFIKASI & ESTIMASI MODEL ARIMA
# ===============================
# Menggunakan auto_arima untuk menemukan orde (p,d,q) terbaik
auto_model = pm.auto_arima(close_data, 
                           start_p=1, start_q=1,
                           test='adf',       # Gunakan ADF test untuk menemukan d
                           max_p=5, max_q=5, # Batas maksimum p dan q
                           m=1,              # Frekuensi data (1 untuk non-musiman)
                           d=None,           # Biarkan model mencari d
                           seasonal=False,   # Tidak ada musiman
                           start_P=0, 
                           D=0, 
                           trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

print("\n===== Ringkasan Model ARIMA Terbaik =====")
print(auto_model.summary())

# ===============================
# CETAK PERSAMAAN MATEMATIS
# ===============================
order = auto_model.order
p, d, q = order
print(f"\nModel ARIMA terbaik adalah ARIMA({p},{d},{q})")

# Fit ulang model dengan statsmodels untuk mendapatkan parameter
model = ARIMA(close_data, order=order)
model_fit = model.fit()

# Ambil koefisien
params = model_fit.params
print("\nKoefisien Model:")
print(params)

# ===============================
# 8. DIAGNOSTIK MODEL
# ===============================
# Plot diagnostik untuk memeriksa sisaan (residuals)
print("\n===== Plot Diagnostik Model =====")
model_fit.plot_diagnostics(figsize=(15, 12))
plt.suptitle(f'Diagnostik untuk Model ARIMA{order}', size=16)
plt.subplots_adjust(top=0.94)
plt.show()

# ===============================
# 9. PREDIKSI 10 HARI KE DEPAN
# ===============================
n_periods = 10
forecast_result = model_fit.get_forecast(steps=n_periods)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int()

print(f"\n===== Prediksi {n_periods} Hari ke Depan =====")
print("Prediksi Harga Rata-rata:")
print(forecast_mean)

# ===============================
# 10. PLOT DATA AKTUAL DAN PREDIKSI
# ===============================
plt.figure(figsize=(14, 7))
# Plot data historis
plt.plot(close_data.index, close_data, label='Data Aktual', color='blue')
# Plot data prediksi
plt.plot(forecast_mean.index, forecast_mean.values, label='Prediksi', color='red', linestyle='--')
# Plot interval kepercayaan
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1], color='pink', alpha=0.5, label='Interval Kepercayaan 95%')

plt.title(f'Harga Penutupan MYOR.JK dan Prediksi {n_periods} Hari ke Depan')
plt.xlabel('Tanggal')
plt.ylabel('Harga')
plt.legend()
plt.show()


# ===============================
# 11. EVALUASI MODEL ARIMA
# ===============================
# Split data (90% training, 10% testing)
split_index = int(len(close_data) * 0.9)
train_data = close_data[:split_index]
test_data = close_data[split_index:]

# Bangun model hanya dengan data training
model_eval = pm.auto_arima(train_data, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')
print(f"\nModel ARIMA untuk evaluasi: {model_eval.order}")

# Prediksi sesuai panjang data testing
predictions = model_eval.predict(n_periods=len(test_data))

# Hitung metrik evaluasi
mae = mean_absolute_error(test_data, predictions)
rmse = np.sqrt(mean_squared_error(test_data, predictions))
mape = mean_absolute_percentage_error(test_data, predictions)

print("\n===== Evaluasi Model ARIMA pada Data Test =====")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2%}")



#################################################################
##                                                             ##
##                   ANALISIS DENGAN PROPHET                   ##
##                                                             ##
#################################################################

print("\n\n" + "="*50)
print(" " * 15 + "ANALISIS DENGAN PROPHET")
print("="*50 + "\n")

# =======================================
# 1. FORMAT DATA SESUAI KEBUTUHAN PROPHET
# =======================================
# Prophet membutuhkan kolom 'ds' (date) dan 'y' (value)
df_prophet = data.reset_index()
df_prophet = df_prophet[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

print("===== Format Data untuk Prophet =====")
print(df_prophet.head())


# =======================================
# 2. SPLIT DATA (90% TRAINING)
# =======================================
split_index_prophet = int(len(df_prophet) * 0.9)
train_df = df_prophet[:split_index_prophet]
test_df = df_prophet[split_index_prophet:]

print(f"\nJumlah data training: {len(train_df)}")
print(f"Jumlah data testing : {len(test_df)}")


# =======================================
# 3. LATIH MODEL PROPHET
# =======================================
model_prophet = Prophet(daily_seasonality=True)
model_prophet.fit(train_df)


# =======================================
# 4. PREDIKSI 100 HARI KE DEPAN
# =======================================
future = model_prophet.make_future_dataframe(periods=100)
forecast = model_prophet.predict(future)

print("\n===== Hasil Prediksi Prophet (beberapa baris terakhir) =====")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())


# =======================================
# 5. VISUALISASI HASIL PREDIKSI
# =======================================
fig1 = model_prophet.plot(forecast)
plt.title("Prediksi Harga Saham MYOR.JK dengan Prophet (100 Hari ke Depan)")
plt.xlabel("Tanggal")
plt.ylabel("Harga")
plt.show()

# Plot komponen tren dan musiman
fig2 = model_prophet.plot_components(forecast)
plt.show()


# =======================================
# 6. EVALUASI MODEL PROPHET
# =======================================
# Buat dataframe future yang hanya mencakup periode data test
future_test = test_df[['ds']]
forecast_test = model_prophet.predict(future_test)

# Ambil nilai aktual dan prediksi
actual_test = test_df['y']
predicted_test = forecast_test['yhat']

# Hitung metrik evaluasi
mae_p = mean_absolute_error(actual_test, predicted_test)
rmse_p = np.sqrt(mean_squared_error(actual_test, predicted_test))
mape_p = mean_absolute_percentage_error(actual_test, predicted_test)
mse_p = mean_squared_error(actual_test, predicted_test)

print("\n===== Evaluasi Model Prophet pada Data Test =====")
print(f"MAE  : {mae_p:.4f}")
print(f"RMSE : {rmse_p:.4f}")
print(f"MAPE : {mape_p:.2%}")
print(f"MSE  : {mse_p:.4f}")