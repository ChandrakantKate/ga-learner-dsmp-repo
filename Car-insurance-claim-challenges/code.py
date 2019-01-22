# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)
print(df.head())
print(df.info())
cols = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']

for col in cols:
    df[col] = df[col].str.replace('$','')
    df[col] = df[col].str.replace(',','')

X = df.drop('CLAIM_FLAG',axis=1)
y = df['CLAIM_FLAG']

count = y.value_counts()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6,test_size=0.3)


# Code ends here


# --------------
# Code starts here
X_train[cols] = X_train[cols].astype('float64')
X_test[cols] = X_test[cols].astype('float64')
print(X_train.isnull().sum())
print(X_test.isnull().sum())
# Code ends here


# --------------
# Code starts here
X_train.dropna(axis=0,subset=['YOJ','OCCUPATION'],inplace=True)
X_test.dropna(axis=0,subset=['YOJ','OCCUPATION'],inplace=True)

y_train = y_train[X_train.index]
y_test = y_test[X_test.index]
cols1 = ['AGE','CAR_AGE','INCOME','HOME_VAL']
X_train[cols1] = X_train[cols1].fillna(X_train[cols1].mean())
X_test[cols1] = X_test[cols1].fillna(X_test[cols1].mean())
# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
le = LabelEncoder()
for col in columns:
    X_train[col] = le.fit_transform(X_train[col].astype('str'))
    X_test[col] = le.transform(X_test[col].astype('str'))

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model = LogisticRegression(random_state=6)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote = SMOTE(random_state=6)
X_train, y_train = smote.fit_sample(X_train,y_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Code ends here


# --------------
# Code Starts here
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test,y_pred)
# Code ends here


