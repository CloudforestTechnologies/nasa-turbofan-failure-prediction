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

def create_baseline_model(dataset_df):
    """
    Creates, trains and returns a linear regression model from training data.
    ======================================

    Input:
        dataset_df (dataframe) - Dataframe containing training dataset.
        
    Output:
	    regr_model (LinearRegression) - Linear Regression model fitted to training data.
    """
    pass

def evaluate_baseline_model(model):
    """
    Orders, prints and returns a list of data values sorted by absolute value.
    ======================================

    Input:
        slopes_array (array) - Array containing normalised column data.
        dataset_df (dataframe) - Dataframe containing parent dataset.
        
    Output:
	    slope_order (array) - An array of data columns, ordered by abs value (highest first).
    """

    pass