import pandas as pd
df = pd.read_csv("/workspaces/2025-an6001a/model/german_credit.csv")

from sklearn.model_selection import train_test_split
from sklearn import linear_model,tree,ensemble
from sklearn.metrics import confusion_matrix,accuracy_score
Y = df['Creditability']
X = df.drop(columns="Creditability")
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)
model = linear_model.LogisticRegression(max_iter=5000)
model.fit(X_train,Y_train)
pred = model.predict(X_test)
print(confusion_matrix(Y_test,pred))
print(accuracy_score(Y_test,pred))

model = tree.DecisionTreeClassifier(random_state=1)
model.fit(X_train,Y_train)
pred = model.predict(X_test)
print(confusion_matrix(Y_test,pred))
print(accuracy_score(Y_test,pred))

pred = model.predict(X_train)
print(confusion_matrix(Y_train,pred))
print(accuracy_score(Y_train,pred))

model = ensemble.RandomForestClassifier(random_state=1)

model.fit(X_train,Y_train)
pred = model.predict(X_test)
print(confusion_matrix(Y_test,pred))
print(accuracy_score(Y_test,pred))

model = ensemble.GradientBoostingClassifier(random_state=1)

model.fit(X_train,Y_train)
pred = model.predict(X_test)
print(confusion_matrix(Y_test,pred))
print(accuracy_score(Y_test,pred))

X = df.loc[:,["Age"]]
Y = df.loc[:,["Creditability"]]
model = ensemble.RandomForestClassifier()
model.fit(X,Y)
import joblib
joblib.dump(model,"german_credit.pkl")