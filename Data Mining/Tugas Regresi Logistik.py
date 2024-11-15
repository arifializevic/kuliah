import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Dataset yang sama seperti sebelumnya
data = {
    'X1': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    'X2': [1, 1, 2, 2, 2, 2, 3, 5, 2, 2, 1, 4, 1, 3, 4, 5, 3, 4, 5, 4, 4, 4, 3, 4, 5, 3, 4, 2, 4, 4, 1, 3, 4, 5, 3, 4, 5, 4, 4],
    'Y':  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

# Menambahkan konstanta
df['intercept'] = 1

# Mendefinisikan variabel independen dan dependen
X = df[['intercept', 'X1', 'X2']]
Y = df['Y']

# Membuat model regresi logistik
model = sm.Logit(Y, X)
result = model.fit()

# Membuat prediksi probabilitas
y_pred_prob = result.predict(X)

# Menghitung ROC curve
fpr, tpr, thresholds = roc_curve(Y, y_pred_prob)

# Menghitung AUC (Area Under Curve)
auc = roc_auc_score(Y, y_pred_prob)

# Membuat plot kurva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})', color='blue')
# Diagonal line for random guess
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
