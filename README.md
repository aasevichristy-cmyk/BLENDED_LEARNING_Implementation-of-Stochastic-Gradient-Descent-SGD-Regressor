# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the car price dataset and perform initial inspection; remove irrelevant columns and convert categorical variables into numerical form using one-hot encoding.
2. Separate the dataset into independent features (X) and target variable (y), and apply standard scaling to normalize the data.
3. Split the scaled data into training and testing sets using an 80:20 ratio.
4. Initialize the Stochastic Gradient Descent (SGD) Regressor and train the model using the training data.
5. Predict car prices on the test data, evaluate the model using MSE, MAE, and R² score, and visualize actual versus predicted values.


## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: Anisha A
RegisterNumber:  212225220009
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("CarPrice_Assignment.csv")
print(data.head())
print(data.info())
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)
X = data.drop('price', axis=1)
y = data['price']
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)
sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)
mse=mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae=mean_absolute_error(y_test,y_pred)
print("Name:Anisha A")
print("Reg. No:212225220009")
print("MSE (Mean Squared Error):", mse)
print("R-squared Score:", r2)
print("MAE (Mean Absolute Error):",mae)
print("Model Coefficients:")
print("Coefficients:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test), max(y_test)],[min(y_test), max(y_test)],color='red')
plt.show()

```

## Output:

 <img width="1302" height="165" alt="Screenshot 2026-02-12 093802" src="https://github.com/user-attachments/assets/7a151c9f-4a9b-4b8f-a257-198bef092687" />
 <img width="890" height="124" alt="Screenshot 2026-02-12 093830" src="https://github.com/user-attachments/assets/31c02f11-61b3-46a7-b94d-9072e9404125" />
 <img width="975" height="235" alt="Screenshot 2026-02-12 093838" src="https://github.com/user-attachments/assets/a64b5a73-eb41-47da-9db0-6c7b06ce123d" />
 <img width="1030" height="579" alt="Screenshot 2026-02-12 093845" src="https://github.com/user-attachments/assets/a91c848a-125b-44ab-bbd1-a6f0814f0170" />





## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
