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

dataset_columns = ( 'Unit Number', 
                    'Time [Cycles]',
                    'Setting 1', 'Setting 2', 'Setting 3',
                    'Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4', 'Sensor 5', 'Sensor 6', 'Sensor 7',
                    'Sensor 8', 'Sensor 9', 'Sensor 10', 'Sensor 11', 'Sensor 12', 'Sensor 13', 'Sensor 14',
                    'Sensor 15', 'Sensor 16', 'Sensor 17', 'Sensor 18', 'Sensor 19', 'Sensor 20', 'Sensor 21' ) 

# Import the files
def import_dataset():

    test_set = pd.read_csv(filename_string, sep = ' ', header = None, names = dataset_columns)

    print(test_set)