# --------------
#Importing header files

import pandas as pd

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split


# Code starts here
data = pd.read_csv(path)
X = data.drop(columns=['customer.id','paid.back.loan'])
y = data['paid.back.loan']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
# Code ends here


# --------------
#Importing header files
import matplotlib.pyplot as plt

# Code starts here
fully_paid = y_train.value_counts()
plt.bar(fully_paid.index,fully_paid)
plt.show()
# Code ends here


# --------------
#Importing header files
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Code starts here
replace = lambda x: x.replace('%','')
X_train['int.rate'] = X_train['int.rate'].apply(replace)
X_train['int.rate'] = pd.to_numeric(X_train['int.rate'])/100

X_test['int.rate'] = pd.to_numeric(X_test['int.rate'].apply(replace))/100
num_df = X_train.select_dtypes(include=[np.number])
cat_df = X_train.select_dtypes(exclude=[np.number])
#print(X_train.info(),num_df,cat_df)
# Code ends here


# --------------
#Importing header files
import seaborn as sns


# Code starts here
cols = list(num_df.columns)
fig, axes = plt.subplots(9,1, figsize=(10,10))

for i in range(9):
    sns.boxplot(x=y_train, y=num_df[cols[i]], ax=axes[i])

# Code ends here


# --------------
# Code starts here
cols = cat_df.columns
fig, axes = plt.subplots(2,2, figsize=(10,10))

for i in range(2):
    for j in range(2):
        sns.countplot(x=X_train[cols[i*2+j]],hue=y_train,ax=axes[i,j])

# Code ends here


# --------------
#Importing header files
from sklearn.tree import DecisionTreeClassifier

# Code starts here
le = LabelEncoder()
for i in range(len(cols)):
    X_train[cols[i]] = le.fit_transform(X_train[cols[i]])
    X_test[cols[i]] = le.transform(X_test[cols[i]])

y_train.replace({'No':0,'Yes':1},inplace=True)
y_test.replace({'No':0,'Yes':1},inplace=True)

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train,y_train)
#y_pred = model.predict(X_test)

acc = model.score(X_test,y_test)
# Code ends here


# --------------
#Importing header files
from sklearn.model_selection import GridSearchCV

#Parameter grid
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}

# Code starts here
model_2 = DecisionTreeClassifier(random_state=0)
p_tree = GridSearchCV(estimator=model_2,param_grid=parameter_grid,cv=5)
p_tree.fit(X_train,y_train)
acc_2 = p_tree.score(X_test,y_test)
# Code ends here


# --------------
#Importing header files

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus

# Code starts here
dot_data = export_graphviz(decision_tree=p_tree.best_estimator_, out_file=None, feature_names=X.columns, filled = True, class_names=['loan_paid_back_yes','loan_paid_back_no'])
graph_big = pydotplus.graph_from_dot_data(dot_data)

# show graph - do not delete/modify the code below this line
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)

plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show() 

# Code ends here


