'''
NASA Turbofan Failure Prediction - Baseline Model

This file supports creating a baseline predictive model. 

'''

###################################
# Module Importations (A - Z Format)
###################################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def create_baseline_model(dataset_df, target_value):
    """
    Creates, trains and returns a linear regression model from training data.
    ======================================

    Input:
        dataset_df (dataframe) - Dataframe containing training dataset.
        training_data (list) - List of training data column headers.
        target_value (str) - Target value for model training.
        
    Output:
	    regr_model (LinearRegression) - Linear Regression model fitted to training data.
    """

    X_array = dataset_df.drop(target_value, axis = 1).values
    y_array = dataset_df[target_value].values

    # Create split between training and test sets.
    print("Splitting data into training and test sets.")
    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size = 0.2, random_state = 0)

    # Initialise the linear regression.
    regr_model = LinearRegression()

    # Train the algorithm on the data.
    print("Training the model.")
    regr_model.fit(X_train, y_train)

    # Evaluate the model.
    y_pred = regr_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    print("Baseline MAE: " + str(mae))

    rmse = mean_squared_error(y_test, y_pred, squared = False)
    print("Baseline MSE: " + str(rmse))

    return regr_model