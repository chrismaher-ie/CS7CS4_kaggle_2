#library imports
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[ ]:


#open, combine and clean csv data
def Get_Clean_Combined_Dataset():

    df = pd.read_csv(r'C:\Users\Me\Documents\College\Maths 4th Year\Machine Learning\Group competition\tcd-ml-1920-group-income-train.csv')

    df.drop_duplicates(subset ="Instance",keep = "first", inplace = True)
    df2 = pd.read_csv(r'C:\Users\Me\Documents\College\Maths 4th Year\Machine Learning\Group competition\tcd-ml-1920-group-income-test.csv')
    df2["Instance"] = df2["Instance"] + 991709
    
    data = pd.concat([df,df2], axis=0)
    data = data.reset_index(drop=True)
    data['Yearly Income in addition to Salary (e.g. Rental Income)'] = data['Yearly Income in addition to Salary (e.g. Rental Income)'].str.replace(' EUR', '').astype(float)
    data = data.replace('#NUM!',np.nan, inplace=False)
    return data


# In[ ]:


#get dataset
data_set = Get_Clean_Combined_Dataset()


# In[ ]:


#target encoder
#catagorical data is replaced with the mean of the means of the feature
#Eg all 'Phd's in University Degree become 93,000
def Target_Encode(df, feature, y_col="Total Yearly Income [EUR]"):
    
    df[feature] = df[feature].replace(np.nan, "Unknown", inplace=False)
    
    means = df.groupby(feature)[y_col].mean()
    
    df[feature] = df[feature].map(means)
    
    return df

#Function to relpace invalid entries in numeric features with the mean walue of the particular feature
def ReplaceNan_Numeric(df,F_Name):
    average = df[F_Name].dropna().mean(axis=0)
    df[F_Name] = df[F_Name].replace(np.nan, average, inplace=False)
    return df


# In[ ]:


#Target encode all the catagorical features
data_set = Target_Encode(data_set, "Profession")
data_set = Target_Encode(data_set, "Satisfation with employer")
data_set = Target_Encode(data_set, "Country") # Country data needs more cleaning
data_set = Target_Encode(data_set, "Gender")
data_set = Target_Encode(data_set, "University Degree")
data_set = Target_Encode(data_set, "Housing Situation")

#Special case for hair colour - from observation it only maters if it is '0'
#Set this to a bool 
data_set["Hair Color"] = np.where(data_set["Hair Color"] =='0', 1, 0)



# In[ ]:


data_set = ReplaceNan_Numeric(data_set, "Year of Record")
data_set = ReplaceNan_Numeric(data_set, "Crime Level in the City of Employement")
data_set["Work Experience in Current Job [years]"] = data_set["Work Experience in Current Job [years]"].astype(float)
data_set = ReplaceNan_Numeric(data_set, "Work Experience in Current Job [years]")
data_set = ReplaceNan_Numeric(data_set, "Age")
data_set = ReplaceNan_Numeric(data_set, "Size of City")
data_set = ReplaceNan_Numeric(data_set, "Wears Glasses")
data_set = ReplaceNan_Numeric(data_set, "Body Height [cm]")
data_set = ReplaceNan_Numeric(data_set, "Yearly Income in addition to Salary (e.g. Rental Income)")

#needs more nan removal
data_set = ReplaceNan_Numeric(data_set, "Country")


# In[ ]:


data_set_1 = data_set.loc[data_set["Year of Record"] <= 1978]
data_set_2 = data_set.loc[data_set["Year of Record"] > 1978]


# In[ ]:




#Select features
X_1 = data_set_1.drop(columns=["Total Yearly Income [EUR]",])
X_2 = data_set_2.drop(columns=["Total Yearly Income [EUR]",])


#Setup target array
y_1 = data_set_1[["Instance","Total Yearly Income [EUR]"]]
y_2 = data_set_2[["Instance","Total Yearly Income [EUR]"]]


# In[ ]:


#Split the data back out to separate sets on key 'Instance'
X_predictions_1 = X_1.loc[X_1["Instance"] > 991709]
X_1 = X_1.loc[X_1["Instance"] <= 991709]         
y_1 = y_1.loc[y_1["Instance"] <= 991709]

#Drop 'Instance' from datasets not that it is nolonger needed as a key
y_pred_instance_1 = X_predictions_1["Instance"]
X_predictions_1 = X_predictions_1.drop("Instance",1)
X_1 = X_1.drop("Instance",1)
y_1 = y_1.drop("Instance",1)


# In[ ]:


#Split the data back out to separate sets on key 'Instance'
X_predictions_2 = X_2.loc[X_2["Instance"] > 991709]
X_2 = X_2.loc[X_2["Instance"] <= 991709]         
y_2 = y_2.loc[y_2["Instance"] <= 991709]

#Drop 'Instance' from datasets not that it is nolonger needed as a key
y_pred_instance_2 = X_predictions_2["Instance"]
X_predictions_2 = X_predictions_2.drop("Instance",1)

X_2 = X_2.drop("Instance",1)
y_2 = y_2.drop("Instance",1)


# In[ ]:


# Split the test and training data
X_train, X_test, y_train, y_test_1 = train_test_split(X_1, y_1, test_size=0.1, random_state=42)

#Setup BayesianRidge regressor
#regr_1 =  lgb.LGBMRegressor()
#
## Train the model using the training sets
#regr_1.fit(X_train, y_train)
#
## Make predictions using the testing set
#y_pred_1 = regr_1.predict(X_test)
#
#
#
#param_test ={'n_iter' : [20,100,500],
#            'num_leaves' : np.arange(5,50, 5), 
#             'min_child_samples': np.arange(100,500, 50), 
#             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
#             }
#
#
#grid_1 = RandomizedSearchCV(estimator=regr_1, param_distributions=param_test, 
#                        scoring='neg_mean_absolute_error', cv=3, refit=True,
#                        random_state=1,  verbose=True)
#
#grid_1.fit(X_train, y_train)
#grid_1.best_params_

#Setup BayesianRidge regressor
regr_1 =  lgb.LGBMRegressor(reg_alpha= 0.1, num_leaves = 45, n_iter= 4000, min_child_samples= 100)

# Train the model using the training sets
regr_1.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_1 = regr_1.predict(X_test)



# In[ ]:


# Split the test and training data
X_train2, X_test2, y_train2, y_test_2 = train_test_split(X_2, y_2, test_size=0.1, random_state=42)

#Setup BayesianRidge regressor
#regr_2 =  lgb.LGBMRegressor()
#
## Train the model using the training sets
#regr_2.fit(X_train, np.sqrt(y_train2))
#
## Make predictions using the testing set
#y_pred_2 = regr_2.predict(X_test2)
#y_pred_2 = np.square(y_pred_2)
#
#param_test ={'n_iter' : [30, 100,500],
#            'num_leaves' : np.arange(1,50, 10), 
#             'min_child_samples': np.arange(100, 300, 50), 
#             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 15],
#             }
#
#
#grid_2 = RandomizedSearchCV(estimator=regr_2, param_distributions=param_test, 
#                        scoring='neg_mean_absolute_error', cv=2, refit=True,
#                        random_state=1,  verbose=True)
#
#grid_2.fit(X_train2, y_train2)
#grid_2.best_params_

#Setup BayesianRidge regressor
regr_2 =  lgb.LGBMRegressor(reg_alpha= 0.1, num_leaves = 45, n_iter= 4000, min_child_samples= 100)

# Train the model using the training sets
regr_2.fit(X_train2, np.sqrt(y_train2))

# Make predictions using the testing set
y_pred_2 = regr_2.predict(X_test2)
y_pred_2 = np.square(y_pred_2)


# In[ ]:


all_pred = np.concatenate((y_pred_1,y_pred_2), axis=None)
all_test = np.concatenate((y_test_1,y_test_2), axis=None)


# In[ ]:




# Print results
print("Mean absolute error: %.2f"
      % mean_absolute_error(all_test, all_pred))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(all_test, all_pred))


# In[ ]:
#Setup BayesianRidge regressor
regr_1 =  lgb.LGBMRegressor(reg_alpha= 0.1, num_leaves = 45, n_iter= 4000, min_child_samples= 100)

# Train the model using the training sets
regr_1.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_1 = regr_1.predict(X_test)


## Model 2
regr_2 =  lgb.LGBMRegressor(reg_alpha= 0.1, num_leaves = 45, n_iter= 4000, min_child_samples= 100)

# Train the model using the training sets
regr_2.fit(X_train2, np.sqrt(y_train2))

# Make predictions using the testing set
y_pred_2 = regr_2.predict(X_test2)
y_pred_2 = np.square(y_pred_2)



# Make predictions on the unlabled dataset
y_pred_1 = regr_1.predict(X_predictions_1)

y_pred_2 = regr_2.predict(X_predictions_2)
y_pred_2 = np.square(y_pred_2)

#all_pred = np.concatenate((y_pred_1,y_pred_2), axis=None)


# In[ ]:


set_1 = pd.DataFrame({'Instance':y_pred_instance_1, 'Total Yearly Income [EUR]':y_pred_1})
set_2 = pd.DataFrame({'Instance':y_pred_instance_2, 'Total Yearly Income [EUR]':y_pred_2})

Results = pd.concat([set_1,set_2], axis=0)

Results = Results.sort_values(by=['Instance'])

#sort
y_results = Results["Total Yearly Income [EUR]"]


# In[ ]:


pred_sample = pd.read_csv("Results6-catboost.csv")
pred_sample = pred_sample["Total Yearly Income [EUR]"]

print("Mean absolute error: %.2f"
      % mean_absolute_error(y_results, pred_sample))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_results, pred_sample))


# In[ ]:


# export data
pd.DataFrame(y_results).to_csv(r'C:\Users\Me\Documents\College\Maths 4th Year\Machine Learning\Group competition\Results7-tuning.1.csv')

#Results.to_csv(r'C:\Users\Me\Documents\College\Maths 4th Year\Machine Learning\Group competition\Results4.1.csv')


# In[ ]:
