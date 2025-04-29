# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Data Preparation 
3. Hypothesis Definition
4. Cost Function
5. Parameter Update Rule
6. Iterative Training
7. Model Evaluation 
8. End

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Keerthana P
RegisterNumber: 212223240069 
*/
```

```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
```

```
data=fetch_california_housing()
print(data)
```
![image](https://github.com/user-attachments/assets/220a9aa5-1685-4705-bb59-3bee56766c7e)


```
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
```
![image](https://github.com/user-attachments/assets/3613c276-93d3-4f78-9c8d-8dea7b58326d)


```
print(df.info())
```
![image](https://github.com/user-attachments/assets/a852afa3-241d-41d5-8307-cc9c725c36c0)

```
print(df.tail())
```
![image](https://github.com/user-attachments/assets/eb6115b2-c8ce-447e-9c60-20e1d346deaa)


```
x=df.drop(columns=['AveOccup','target'])
x.info()
print(x.shape)
```
![image](https://github.com/user-attachments/assets/8b88a740-0078-47a4-83e0-5292cc96c88f)

```
y=df[['AveOccup','target']]
y.info()
print(y.shape)
```

![image](https://github.com/user-attachments/assets/cfde1694-6fda-4ef1-befd-e41a2e327dd5)

```
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=11)
x.head()
```

![image](https://github.com/user-attachments/assets/a9323e61-f33c-4d91-be4f-0259010cc009)

```
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
```

![image](https://github.com/user-attachments/assets/5aee36cd-cb41-4a26-8edb-84ef7db973be)

```
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)
print(x_train)
```

![image](https://github.com/user-attachments/assets/b08f0c1e-35d4-45c0-b764-195902c7db6c)

```
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)
```

![image](https://github.com/user-attachments/assets/f386d6fb-2b5b-4fe9-ab6e-5aeca3c27375)
```
y_pred=multi_output_sgd.predict(x_test)
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)
```

![image](https://github.com/user-attachments/assets/d1299dba-70f8-4f9c-8294-d8b2f91cf6d6)

```
print("\nPredictions:\n", y_pred[:5])
```

![image](https://github.com/user-attachments/assets/89e6d3d2-2946-452b-acb9-244509dd6764)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
