import pandas as pd
df = pd.read_csv("/workspaces/2025-an6001a/model/DBS_SingDollar.csv")
X = df.loc[:,["SGD"]]
Y = df.loc[:,["DBS"]]

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
model = linear_model.LinearRegression()
model.fit(X,Y)

#print(model.intercept_)

import joblib
joblib.dump(model,"/workspaces/2025-an6001a/dbs.pkl")
