'''
NASA Turbofan Failure Prediction - Baseline Model

This file supports creating a baseline predictive model. 

'''

###################################
# Module Importations (A - Z Format)
###################################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
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

    # Create split between training and test sets.
    print("Splitting data into training and test sets.")
    train, test = train_test_split(dataset_df, test_size = 0.2, random_state = 0)

    # Initialise the linear regression.
    regr_model = LinearRegression()

    # Create training set.
    X_train = train
    y_train = train[target_value]

    # Train the algorithm on the data.
    print("Training the model.")
    regr_model.fit(X_train, y_train)

    # Evaluate the model.
    y_test = test[target_value]
    y_pred = regr_model.predict(test)

    mae = mean_absolute_error(y_test, y_pred)
    print("Baseline MAE: " + str(mae))

    return regr_model