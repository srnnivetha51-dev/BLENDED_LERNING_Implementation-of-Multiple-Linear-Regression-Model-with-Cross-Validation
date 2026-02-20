# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Import required libraries (pandas, sklearn, matplotlib).
3. Load the dataset from CSV file.
4. Remove unnecessary columns (car_ID, CarName).
5. Convert categorical variables into dummy variables.
6. Separate features (X) and target variable (price).
7. Split data into training and testing sets.
8. Train Linear Regression model using training data.
9. Perform 5-fold cross-validation and evaluate test performance (MSE, MAE, R²).
10. Plot Actual vs Predicted prices and Stop. 

## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.

import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt

data=pd.read_csv("CarPrice_Assignment.csv")
print(data.head())

data=data.drop(['car_ID','CarName'],axis=1)
data=pd.get_dummies(data,drop_first=True)
print(data.head())

x=data.drop('price',axis=1)
y=data['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

print("Name: S R NIVEDHITHA ")
print("Reg. No:212225240102")
print("\n=== Cross-Validation ===")
cv_scores=cross_val_score(model,x,y,cv=5)
print("Fold R^2 scores:",[f"{score:.4f}"for score in cv_scores])
print("Average R^2: {cv_scores.mean():.4f}")

y_pred=model.predict(x_test)
print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.2f}")
print(f"R^2: {r2_score(y_test,y_pred):.4f}")

plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual Price")
plt.grid(True)
plt.show()

Developed by: S R NIVEDHITHA
RegisterNumber: 25000724 
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![alt text](<Screenshot 2026-02-20 140731.png>)
![alt text](<Screenshot 2026-02-20 140752.png>)
![alt text](<Screenshot 2026-02-20 140807.png>)
![alt text](<Screenshot 2026-02-20 140824.png>)
![alt text](<Screenshot 2026-02-20 140843.png>)


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
