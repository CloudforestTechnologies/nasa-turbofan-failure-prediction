'''
NASA Turbofan Failure Prediction - Random Forest

This file supports training and evaluating a random forest regression model. 

'''

###################################
# Module Importations (A - Z Format)
###################################
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

def train_random_forest_model(dataset_df, target_value):
    """
    Creates, trains and returns a random forest model from training data.
    ======================================

    Input:
        dataset_df (dataframe) - Dataframe containing training dataset.
        target_value (str) - Target value for model training.
        
    Output:
	    rf_model (RandomForestRegression) - Random Forest Regression model fitted to training data.
    """

    X_array = dataset_df.drop(target_value, axis = 1).values
    y_array = dataset_df[target_value].values

    # Create split between training and test sets.
    print("[Random Forest Regression] Splitting data into training and test sets.")
    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size = 0.2, random_state = 0)

    # Initialise the linear regression.
    rf_model = RandomForestRegressor()

    # Train the algorithm on the data.
    print("[Random Forest Regression] Training the model.")
    rf_model.fit(X_train, y_train)

    # Evaluate the model.
    y_pred = rf_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    print("Random Forest Regression MAE: " + str(mae))

    rmse = mean_squared_error(y_test, y_pred, squared = False)
    print("Random Forest Regression MSE: " + str(rmse))

    return rf_model