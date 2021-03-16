'''
Constants

Provides store of constants for use with this project

'''

# Module Imports
import os

def return_project_code():
    return 'PM'

def return_fig_save_path():
    return r'D:/Developer Area/nasa-turbofan-failure-prediction/Reports/Figures'

def return_data_pickle_path():
    return r'D:/Developer Area/nasa-turbofan-failure-prediction/Data/Interim'

def return_model_save_path():
    return r'D:/Developer Area/nasa-turbofan-failure-prediction/Models'

def return_tensorboard_log_path():

    root = os.getcwd()
    branch = r"Models\TensorBoard"

    path = os.path.join(root, branch)
    
    return path