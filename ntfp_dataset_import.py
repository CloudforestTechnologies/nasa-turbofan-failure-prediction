'''
NASA Turbofan Failure Prediction - Dataset Import

This file supports datset importation and visualisation.

'''

###################################
# Module Importations (A - Z Format)
###################################
import pandas as pd

# Constants
filename_string = r'C:\Developer\PMetcalf\nasa_turbofan_failure_prediction\Raw Data\test_FD001.txt'

# Import the files
def import_dataset():

    test_set = pd.read_csv(filename_string, sep = ' ', header = None)

    print(test_set)