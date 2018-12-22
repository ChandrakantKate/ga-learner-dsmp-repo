# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here
dataset = pd.read_csv(path)
print(dataset.head())
# read the dataset

#dataset['Soil_Type15'].value_counts()

# look at the first five columns


# Check if there's any column which is not useful and remove it like the column id
dataset.drop('Id',axis=1,inplace=True)

# check the statistical description
dataset.describe()


# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols = dataset.columns

#number of attributes (exclude target)
size = len(cols)

#x-axis has target attribute to distinguish between classes
x = cols[-1]

#y-axis shows values of an attribute
y = cols[:-1]

#Plot violin for all attributes
#for 
#ax = sns.violinplot(y,x)







# --------------
import numpy as np
import seaborn as sns
threshold = 0.5
num_features = 10

subset_train = dataset.iloc[:,:num_features]
cols = subset_train.columns
data_corr = subset_train.corr()
sns.heatmap(data_corr)
#data_corr = np.tril(data_corr)
#print(data_corr)
mask1 = data_corr > threshold
mask2 = data_corr > -threshold
mask3 = data_corr != 1
corr_var_list = data_corr[mask1 & mask2 & mask3]
corr_var_list.dropna(how='all',inplace=True)
print(corr_var_list)
#s_corr_list = []
n = data_corr.shape[0]
drop_cols = set()
bol_df = data_corr.isnull()
for i in range(0,data_corr.shape[1]):
    for j in range(0,i+1):
        if ~(bol_df.iloc[i][j]):
            drop_cols.add((cols[i],cols[j]))

#s_corr_list = sorted(data_corr,reverse=True,key=abs)   
s_corr_list = data_corr.unstack().drop(labels=drop_cols)
print(s_corr_list)
# Sort the list showing higher ones first 


#Print correlations and column names




# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler

# Identify the unnecessary columns and remove it 
scaler = StandardScaler()
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)
X,Y = dataset.drop('Cover_Type',axis=1), dataset['Cover_Type']
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size=0.2,random_state=0)

# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.
num_cols = X_train.iloc[:,:10].columns
X_train_temp = pd.DataFrame(scaler.fit_transform(X_train[num_cols]))
X_test_temp = pd.DataFrame(scaler.fit_transform(X_test[num_cols]))
print(X_train_temp)


#Standardized
#Apply transform only for non-categorical data
X_train1 = pd.concat([X_train_temp,X_train],axis=1)
X_test1 = pd.concat([X_test_temp,X_test],axis=1)

#Concatenate non-categorical data and categorical
scaled_features_train_df = pd.DataFrame(X_train1,columns=X_train.columns, index=X_train.index)
scaled_features_test_df = pd.DataFrame(X_test1,columns=X_test.columns, index=X_test.index)



# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:
skb = SelectPercentile(f_classif,percentile=20)
predictors = skb.fit_transform(X_train,Y_train)
X_train1.dropna(axis=0,inplace=True)
cols = X_train.columns
scores = skb.scores_
#top_percentile = scores/scores.max()
top_k_index = np.argsort(scores)[::-1]
top_k_predictors = [cols[i] for i in np.argsort(scores)[::-1]]
top_k_predictors = top_k_predictors[:11] 
print(top_k_index)





# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

clf = OneVsRestClassifier(LogisticRegression())
clf1=OneVsRestClassifier(LogisticRegression())
model_fit_all_features = clf.fit(X_train , Y_train)
predictions_all_features=clf.predict(X_test)
score_all_features= accuracy_score(Y_test,predictions_all_features )
#print(len(scaled_features_train_df.columns))
#print(len(skb.get_support()))
print(scaled_features_train_df.columns[skb.get_support()])
#print(X_new.head())

X_new = scaled_features_train_df.loc[:,skb.get_support()]
X_test_new=scaled_features_test_df.loc[:,skb.get_support()]
#print(y_test)
model_fit_top_features  =clf1.fit(X_new , Y_train)
predictions_top_features=clf1.predict(X_test_new)
#print(predictions_top_features)
score_top_features= accuracy_score(Y_test,predictions_top_features )



