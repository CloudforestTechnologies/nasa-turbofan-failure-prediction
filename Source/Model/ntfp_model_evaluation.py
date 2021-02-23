'''
model_evaluation
Helper routines for evaluating model performance
'''

# Module Importations
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def return_model_evaluation_stats(y_true, y_pred):
    """
    Return model evaluation metrics.
    ======================================
    Input:
        y_true (array) - Untouched evaluation label data.
        y_pred (array) - Model prediction based on evaluation data.
    Output:
        rmse_eval (float) - Root Mean Squared Error.
        mae_eval (float) - Mean Absolute Error.
        r2_eval (float) - R2 Error. 
    """

    # Calculate performance metrics
    rmse_eval = evaluate_rmse(y_true, y_pred)
    mae_eval = evaluate_mae(y_true, y_pred)    
    r2_eval = evaluate_r2(y_true, y_pred)

    # return results
    return rmse_eval, mae_eval, r2_eval

def evaluate_model(model_name, y_true, y_pred):
    """
    Evaluate model using series of metrics.
    ======================================
    Input:
        model_name (string) - Name of model.
        y_true (array) - Untouched evaluation label data.
        y_pred (array) - Model prediction based on evaluation data.
    Output:
        None
    """

    # Calculate performance metrics
    rmse_eval = evaluate_rmse(y_true, y_pred)
    mae_eval = evaluate_mae(y_true, y_pred)    
    r2_eval = evaluate_r2(y_true, y_pred)

    # Print results
    print_evaluation(model_name, mae_eval, rmse_eval, r2_eval)

def evaluate_rmse(y_true, y_pred):
    """
    Evaluate root mean squared error.
    ======================================
    Input:
        y_true (array) - Untouched evaluation label data.
        y_pred (array) - Model prediction based on evaluation data.
    Output:
        rmse_eval (float) - Calculated root mean squared error.
    """

    mse_eval = mean_squared_error(y_true, y_pred)

    rmse_eval = np.sqrt(mse_eval)

    return rmse_eval

def evaluate_mae(y_true, y_pred):
    """
    Evaluate mean absolute error.
    ======================================
    Input:
        y_true (array) - Untouched evaluation label data.
        y_pred (array) - Model prediction based on evaluation data.
    Output:
        mae_eval (float) - Calculated mean absolute error.
    """

    mae_eval = mean_absolute_error(y_true, y_pred)

    return mae_eval

def evaluate_r2(y_true, y_pred):
    """
    Evaluate r2 performance metric.
    ======================================
    Input:
        y_true (array) - Untouched evaluation label data.
        y_pred (array) - Model prediction based on evaluation data.
    Output:
        r2_eval (float) - Calculated r2 performance metric.
    """

    r2_eval = r2_score(y_true, y_pred)

    return r2_eval

def print_evaluation(model_name, mae_eval, rmse_eval, r2_eval):
    """
    Print performance metrics of associated model.
    ======================================
    Input:
        model_name (string) - Name of model.
        rmse_eval (float) - Calculated root mean squared error.
        mae_eval (float) - Untouched evaluation label data.
        r2_eval (float) - Calculated r2 performance metric.
    Output:
        None.
    """

    print(model_name, "rmse (Eval):", rmse_eval)
    print(model_name, "mae (Eval):", mae_eval)
    print(model_name, "r2 (Eval):", r2_eval)