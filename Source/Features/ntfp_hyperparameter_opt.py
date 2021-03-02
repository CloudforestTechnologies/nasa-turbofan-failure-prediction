'''
hyperparameter_opt

This file supports hyperparameter optimisation for ML model development.

'''

# Module importations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def perform_gridsearch(model_name, model, training_data, label_data):
    """Perform Hyperparameter Gridsearch Optimisation
    ======================================
    Explores model hyperparameter optimisation through cross-validation.
    
    Args:
        model_name (string) - Name of model to optimised.
        model (model) - Model to be trained and optimised.
        training_data (dataframe) - Model training input data.
        label_data (dataframe) - Model training target data. 
        
    Returns:
        best_params (dict) - Best parameters determined by optimisation.
        best_est (model) - Best estimator with optimised hyperparameters.
        best_score (float64) - Best score produced by best model.
    """
    
    # Exploratory hyperparameters
    param_grid = [ {'n_estimators': [3, 10, 30, 50, 100], 'max_features': [2, 4, 6, 8, 10, 12]},
                    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
                    ]

    # Set grid search parameters
    grid_search = GridSearchCV(model, param_grid, cv = 5, scoring = 'neg_mean_squared_error', return_train_score = True)

    # Perform grid search
    grid_search.fit(training_data, label_data)

    # Identify optimal parameters, model and score
    best_params = grid_search.best_params_
    print(model_name, "- Best Params:", best_params)

    best_est = grid_search.best_estimator_
    print(model_name, "- Best Estimator:", best_est)

    best_score = np.sqrt(-1 * grid_search.best_score_)
    print(model_name, "- Best Score:", best_score)

    return best_params, best_est, best_score
