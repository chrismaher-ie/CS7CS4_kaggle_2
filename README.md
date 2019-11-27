# CS7CS4 Group project - Income predictor competition 

# Team 41

#### Best submmision source code:
2_models_with_tuning_(Top_submission).py

#### Best submission predictions:
Results7-tuning.1.csv

#### Model
For our best submission we had two LGBMRegressor models 
 - One model to predict records from before 1978
 - One model to predict records from after 1978
     - *Note the second model trained on the square root of income and then predictions were squared afterwards*
 

 #### Data processing
 Catagorical features were target encoded (label replaced by mean income of label)
 Numerical data was cleaned and NaN values replaced with the mean of the feature
