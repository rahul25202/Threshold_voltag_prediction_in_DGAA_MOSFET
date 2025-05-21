
# !pip install xgboost

import xgboost as xgb
import pandas as pd

df = pd.read_csv("dgaa_mosfet_threshold_voltage_dataset.csv")

df.head()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, r2_score

X = df.drop("Threshold_Voltage_V",axis = 1)
y = df["Threshold_Voltage_V"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=100)

train_dmatrix = xgb.DMatrix(data = X_train,label = y_train)
test_dmatrix = xgb.DMatrix(data = X_test,label = y_test)

X = df.drop("Threshold_Voltage_V",axis = 1)
y = df["Threshold_Voltage_V"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=100)

train_dmatrix = xgb.DMatrix(data = X_train,label = y_train)
test_dmatrix = xgb.DMatrix(data = X_test,label = y_test)

param = {'booster':"gblinear" , 'objective' : 'reg:linear'}
xgbr = xgb.train(params = param, dtrain = train_dmatrix,num_boost_round = 10)
y_pred = xgbr.predict(test_dmatrix,output_margin = True)

rmse = mse(y_test,y_pred)**0.5
r2 = r2_score(y_test,y_pred)

print(f'the r2 score is {r2}')
print(f'the rmse score is {rmse}')

