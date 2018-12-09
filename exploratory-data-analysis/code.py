# --------------
#Code starts here
data['Price'].value_counts()

data['Price'] = data['Price'].str.replace('$','').apply(float)

graph = sns.regplot(x="Price", y="Rating" , data=data)
graph.set_title('Rating vs Price [RegPlot]')


#Code ends here


# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data = pd.read_csv(path)
#sns.distplot(data['Rating'])

data = data[data['Rating']<6]
plt.hist(data['Rating'],bins=20)
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
print(data['Installs'].head())

data['Installs'] = data['Installs'].str.replace(',','')
data['Installs'] = data['Installs'].str.replace('+','')
data['Installs'] = data['Installs'].apply(int)
print(data['Installs'].head())

le = LabelEncoder()
data['Installs']=le.fit_transform(data['Installs'])
#le.transform(data['Installs'])

graph = sns.regplot(x="Installs", y="Rating" , data=data)
graph.set_title('Rating vs Installs [RegPlot]')
#Code ends here



# --------------
# code starts here
total_null = data.isnull().sum()

percent_null = total_null/data.isnull().count()

missing_data = pd.concat([total_null,percent_null],keys=['Total','Percent'],axis=1)

print(missing_data) 

data.dropna(how='any',inplace=True)

total_null_1 = data.isnull().sum()

percent_null_1 = total_null_1/data.isnull().count()

missing_data_1 = pd.concat([total_null_1,percent_null_1],keys=['Total','Percent'],axis=1)
# code ends here


# --------------

#Code starts here
graph = sns.catplot(x="Category",y="Rating",data=data, kind="box" , height = 10)
graph.set_xticklabels(rotation=90)
graph.set_titles('Rating vs Category [BoxPlot]')

#Code ends here


# --------------

#Code starts here

def split_Genres(s):
    return s.split(';')[0]

print(data['Genres'].nunique())

data['Genres'] = data['Genres'].apply(split_Genres)

gr_mean = data.groupby(['Genres'],as_index=False)['Genres','Rating'].mean()

print(gr_mean.describe())

gr_mean = gr_mean.sort_values(by='Rating')

print(gr_mean.iloc[0],gr_mean.iloc[-1],sep='\n')
#Code ends here


# --------------

#Code starts here
data['Last Updated'] = pd.to_datetime(data['Last Updated'])
print(data['Last Updated'].head())

max_date = max(data['Last Updated'])

data['Last Updated Days'] = (max_date - data['Last Updated']).dt.days

graph = sns.regplot('Last Updated Days','Rating',data=data)
graph.set_title('Rating vs Last Updated [RegPlot]')
#Code ends here


