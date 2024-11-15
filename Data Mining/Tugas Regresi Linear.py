from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
salary = pd.read_csv('dataset\salary.csv')

# Step 3 : define target (y) and features (X)
salary.columns
y = salary['Salary']
X = salary[['Experience Years']]
# print(y)

# Step 4 : train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=2529)

# check shape of train and test sample
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Step 5 : select model
model = LinearRegression()

# Step 6 : train or fit model
model.fit(X_train, y_train)

model.intercept_
model.coef_

# predict model
y_pred = model.predict(X_test)

print(mean_absolute_error(y_test, y_pred))
print(mean_absolute_percentage_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
