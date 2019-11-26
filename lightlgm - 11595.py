import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


## Data 
train =pd.read_csv(r'C:\Users\Me\Documents\College\Maths 4th Year\Machine Learning\Group competition\tcd-ml-1920-group-income-train.csv')
test = pd.read_csv(r'C:\Users\Me\Documents\College\Maths 4th Year\Machine Learning\Group competition\tcd-ml-1920-group-income-test.csv')

train['train'] =1
test['test'] = 0

##Useful functions
#income['Housing Situation'].unique()
#income.isna().any()

##Data Cleaning year of record
recyear_fill = np.where(train['Total Yearly Income [EUR]'] > 50000 , '2010' , '1970' )
train['Year of Record'] =  train['Year of Record'].fillna(-train['Total Yearly Income [EUR]'] )
train['Year of Record'] =np.where(train['Year of Record'] < -50000, 2010, train['Year of Record'])
train['Year of Record'] =np.where(train['Year of Record'] < 0, 1970, train['Year of Record'])

income =  pd.concat([train, test])
income.drop_duplicates(inplace=True)

med_yor = np.nanmedian(income['Year of Record'])
income['Year of Record'] = income['Year of Record'].fillna(med_yor)

##Target encoding 
prof_means = income.groupby('Profession')['Total Yearly Income [EUR]'].mean()
income['Profession'] = income['Profession'].map(prof_means)
med_income= np.nanmedian(income['Profession'])
income['Profession']= income['Profession'].fillna(med_income)

country_means = income.groupby('Country')['Total Yearly Income [EUR]'].mean()
income['Country'] = income['Country'].map(country_means)
med_income_country= np.nanmedian(income['Country'])
income['Country']= income['Country'].fillna(med_income_country)

#Data Cleaning dopping cols
income.drop(['Age'], axis=1, inplace = True)
income.drop(['Instance'], axis=1, inplace = True)
income.drop(['Wears Glasses'], axis=1, inplace = True)


##Data cleaning : Gender
income.Gender[income.Gender == 'f' ] = 'female'
income.Gender[income.Gender == 'other'  ] = 0
income.Gender[income.Gender == 'unknown' ] = 0
income.Gender[income.Gender == '0' ] = 0
income.Gender= income.Gender.fillna(0)

income = pd.concat([income ,pd.get_dummies(income['Gender'], prefix='Gender' , drop_first=True)],axis=1)
income.drop(['Gender'], axis =1, inplace = True)

##Data cleaning: Hair
income['Hair Color'][income['Hair Color'] == 'Unknown' ] = 'Other' 
income['Hair Color'][income['Hair Color'] == '0' ] = 'Other'
income['Hair Color']= income['Hair Color'].fillna('Other')

income = pd.concat([income ,pd.get_dummies(income['Hair Color'], prefix='Hair_Col',  drop_first=True )],axis=1)
income.drop(['Hair Color'], axis =1, inplace = True)


## Data cleaning : University Degree
income['University Degree'][income['University Degree'] == '0' ] = 'No'
income['University Degree']= income['University Degree'].fillna('No')

income = pd.concat([income ,pd.get_dummies(income['University Degree'], prefix='Degree',  drop_first=True )],axis=1)
income.drop(['University Degree'], axis =1, inplace = True)

##Dropping nas
#income.dropna(axis=0, inplace=True)

##Data cleaning : Housing Situation
income['Housing Situation'][income['Housing Situation'] == 'nA' ] = '0'
income['Housing Situation'][income['Housing Situation'] == '0' ] = 0
income = pd.concat([income ,pd.get_dummies(income['Housing Situation'], prefix='House',  drop_first=True )],axis=1)
income.drop(['Housing Situation'], axis =1, inplace = True)


## Data cleaning Additional income 
income['Yearly Income in addition to Salary (e.g. Rental Income)'] = income['Yearly Income in addition to Salary (e.g. Rental Income)'].astype(str).str.extract('(\d*\.?\d*)', expand=False).astype(float)


##Data cleaning: Satistfaction
income = pd.concat([income ,pd.get_dummies(income['Satisfation with employer'], prefix='Satisf',  drop_first=True )],axis=1)
income.drop(['Satisfation with employer'], axis =1, inplace = True)


##Data cleaning : Work experience
income['Work Experience in Current Job [years]'][income['Work Experience in Current Job [years]'] == '#NUM!' ] = 22
income['Work Experience in Current Job [years]'] = income['Work Experience in Current Job [years]'].astype(float)
#plt.scatter( income['Work Experience in Current Job [years]'],income['Total Yearly Income [EUR]'])

##Dropping hair colour and height 
income.drop(['Body Height [cm]','Hair_Col_Blond','Hair_Col_Brown', 'Hair_Col_Other', 'Hair_Col_Red'], axis = 1, inplace = True)



## Splitting Data
income_train = income[income['train']==1]
income_test = income[income['test']==0]
income_train.drop(['train'], axis=1, inplace=True)
income_train.drop(['test'], axis=1, inplace=True)
income_test.drop(['train'], axis=1, inplace=True)
income_test.drop(['test'], axis=1, inplace=True)


##### Light GMB
X = income_train.loc[:, income_train.columns !='Total Yearly Income [EUR]']
#Y = np.sqrt(income_train.loc[:, 'Total Yearly Income [EUR]'])
inc = np.subtract(income_train['Total Yearly Income [EUR]'], income_train['Yearly Income in addition to Salary (e.g. Rental Income)'])
Y= np.sqrt(inc)

model = lgb.LGBMRegressor()
model.fit(X,Y)
#cv_predict_LGBM = cross_val_score(model, X, Y, cv=5)

## predictions
Results = pd.DataFrame()
Results['Income'] =  model.predict(income_test.loc[:, income_test.columns != 'Total Yearly Income [EUR]'])
Results['Income'] = np.square(Results['Income'])
Results['Income'] = np.add(income_test['Yearly Income in addition to Salary (e.g. Rental Income)'], Results['Income'])
Results.to_csv(r'C:\Users\Me\Documents\College\Maths 4th Year\Machine Learning\Group competition\Results3.csv')





###Local check MAE = 10906
### log 10808
### sqrt 10336
#X = income_train.loc[:, income_train.columns !='Total Yearly Income [EUR]']
##Y = np.sqrt(inc)
#Y = np.sqrt(income_train.loc[:, 'Total Yearly Income [EUR]'])
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
#
#
#
#
#model = lgb.LGBMRegressor()
#model.fit(X_train, Y_train)
#Y_pred = model.predict(X_test)
#
#print(metrics.mean_absolute_error(np.square(Y_test), np.square(Y_pred)))
#


