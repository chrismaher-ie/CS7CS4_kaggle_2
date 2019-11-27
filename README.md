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
 
#### Tuning 
The LGBMReggors were tuned using a randomised search with 3 fold cross validation. Due to time constraints, only one was fine tuned and these parameters were then used for the other model too. This is not ideal, as the second model would have different data and a different model. If we had more time and computational power, we would have extended the parameter grid to a larger range and run this for both models seperately with a 5 fold cv. 

#### Data processing
Catagorical features were target encoded (label replaced by mean income of label)
Numerical data was cleaned and NaN values replaced with the mean of the feature
