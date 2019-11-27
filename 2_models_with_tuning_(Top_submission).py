
# In[ ]:
#library imports
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# In[ ]:


#open, combine and clean csv data
def Get_Clean_Combined_Dataset():

    df = pd.read_csv('tcd-ml-1920-group-income-train.csv')

    df.drop_duplicates(subset ="Instance",keep = "first", inplace = True)
    df2 = pd.read_csv('tcd-ml-1920-group-income-test.csv')
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

#Split the dataset in two sections for records before 1978 and records after 1978
#This is so different models can be used for each data_set
data_set_1 = data_set.loc[data_set["Year of Record"] <= 1978]
data_set_2 = data_set.loc[data_set["Year of Record"] > 1978]


# In[ ]:




#Setup features Matrix for both data sets
X_1 = data_set_1.drop(columns=["Total Yearly Income [EUR]",])
X_2 = data_set_2.drop(columns=["Total Yearly Income [EUR]",])


#Setup target array for both data sets (Instance used as a key for reorgansing data and is removed later)
y_1 = data_set_1[["Instance","Total Yearly Income [EUR]"]]
y_2 = data_set_2[["Instance","Total Yearly Income [EUR]"]]


# In[ ]:

#Data set 1
#Split the data back out to labeled and unlabed data on key 'Instance'

X_predictions_1 = X_1.loc[X_1["Instance"] > 991709]
X_1 = X_1.loc[X_1["Instance"] <= 991709]         
y_1 = y_1.loc[y_1["Instance"] <= 991709]

# store the order of the records so the two models can be combined later
y_pred_instance_1 = X_predictions_1["Instance"]

#Drop 'Instance' from datasets not that it is nolonger needed as a key

X_predictions_1 = X_predictions_1.drop("Instance",1)
X_1 = X_1.drop("Instance",1)
y_1 = y_1.drop("Instance",1)


# In[ ]:

#Data set 2

#Split the data back out to labeled and unlabed data on key 'Instance'
X_predictions_2 = X_2.loc[X_2["Instance"] > 991709]
X_2 = X_2.loc[X_2["Instance"] <= 991709]         
y_2 = y_2.loc[y_2["Instance"] <= 991709]

# store the order of the records so the two models can be combined later
y_pred_instance_2 = X_predictions_2["Instance"]

#Drop 'Instance' from datasets not that it is nolonger needed as a key

X_predictions_2 = X_predictions_2.drop("Instance",1)

X_2 = X_2.drop("Instance",1)
y_2 = y_2.drop("Instance",1)

# In[ ]:

#Parameter tuning using randomised search with cv=3
param_test ={'n_iter' : np.arange(500,4000, 500) ,
            'num_leaves' : np.arange(5,50, 5), 
             'min_child_samples': np.arange(100,500, 50), 
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             }


grid_1 = RandomizedSearchCV(estimator=regr_1, param_distributions=param_test, 
                            scoring='neg_mean_absolute_error', cv=3, refit=True,
                            random_state=1,  verbose=True)

grid_1.fit(X_train, y_train)
grid_1.best_params_

# In[ ]:

#Model 1 on labeled Data set 1

# Split the test and training data
X_train, X_test, y_train, y_test_1 = train_test_split(X_1, y_1, test_size=0.1, random_state=42)


#Setup LGBMRegressor
regr_1 =  lgb.LGBMRegressor(reg_alpha= 0.1, num_leaves = 45, n_iter= 4000, min_child_samples= 100)

# Train the model using the training sets
regr_1.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_1 = regr_1.predict(X_test)



# In[ ]:

#Model 2 on labeled Data set 2

# Split the test and training data
X_train2, X_test2, y_train2, y_test_2 = train_test_split(X_2, y_2, test_size=0.1, random_state=42)

#Setup LGBMRegressor
regr_2 =  lgb.LGBMRegressor(reg_alpha= 0.1, num_leaves = 45, n_iter= 4000, min_child_samples= 100)

# Train the model using the training sets
# note the square root on the y values
regr_2.fit(X_train2, np.sqrt(y_train2))

# Make predictions using the testing set
y_pred_2 = regr_2.predict(X_test2)
#predictions need to be squared back to undo sqrt
y_pred_2 = np.square(y_pred_2)


# In[ ]:

#combine the y predictions and y true values from both models
all_pred = np.concatenate((y_pred_1,y_pred_2), axis=None)
all_test = np.concatenate((y_test_1,y_test_2), axis=None)
#NB as the labeled data is ordered on year no sorting is needed here, this is not true of the unlabed data

# Print results
print("Mean absolute error: %.2f"
      % mean_absolute_error(all_test, all_pred))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(all_test, all_pred))

# In[ ]:


# Make predictions on the unlabled dataset

#Model 1 on unlabeled Data set 1
y_pred_1 = regr_1.predict(X_predictions_1)

#Model 2 on unlabeled Data set 2
y_pred_2 = regr_2.predict(X_predictions_2)
#note squaring of prefictions
y_pred_2 = np.square(y_pred_2)


# In[ ]:

#setup dataframes with the two prediction sets and the corespoding instance sets
set_1 = pd.DataFrame({'Instance':y_pred_instance_1, 'Total Yearly Income [EUR]':y_pred_1})
set_2 = pd.DataFrame({'Instance':y_pred_instance_2, 'Total Yearly Income [EUR]':y_pred_2})
#combine the sets
Results = pd.concat([set_1,set_2], axis=0)

#sort dataframe on instance to reorder the predictions
Results = Results.sort_values(by=['Instance'])

y_results = Results["Total Yearly Income [EUR]"]


# In[ ]:


# export data
pd.DataFrame(y_results).to_csv('Results7-tuning.1.csv')
