'''
NASA Turbofan Failure Prediction - Dataset Import

This file supports dataset importation and exploratory data analysis.

'''

###################################
# Module Importations (A - Z Format)
###################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Constants
dataset_columns = ('Engine', 'Cycles',
                    'Set-1', 'Set-2', 'Set-3', 
                    'Sn_01', 'Sn_02', 'Sn_03', 'Sn_04', 'Sn_05', 'Sn_06', 'Sn_07',
                    'Sn_08', 'Sn_09', 'Sn_10', 'Sn_11', 'Sn_12', 'n_13', 'Sn_14',
                    'Sn_15', 'Sn_16', 'Sn_17', 'Sn_18', 'Sn_19', 'Sn_20', 'Sn_21' ) 

filename_string = r'C:\Developer\PMetcalf\nasa_turbofan_failure_prediction\data\train_FD001.txt'

plot_storage_string = r'C:\Users\ASUS-PC\OneDrive\Cloudforest Technologies\M. Projects\Pink Moss\WP2.0 - NASA Turbofan Failure Prediction\Images'

def import_dataset():
    """
    Import the dataset as a dataframe, adding column names.
    ======================================

    Input:
        None.

    Output:
        raw_data_df (dataframe) - Raw data as dataframe, with column names.
    """

    # Import the raw data as an array.
    raw_data_array = np.loadtxt(filename_string, delimiter = ' ', usecols = range(26))

    # Convert array into dataframe, to add column names and create index.
    raw_data_df = pd.DataFrame(raw_data_array, index = None, columns = dataset_columns)

    return raw_data_df

def peek_dataset(data_df):
    """
    Prints a series of data about the supplied dataframe.
    ======================================

    Input:
        data_df (dataframe) - Dataframe with data to be visualised.

    Output:
        None.
    """

    # Visualise dataframe information.
    print(data_df.head(5))

    print(data_df.info())

    print(data_df.describe())

def visualise_sensor_correlation_all_engine(data_df):
    """
    Plot and save the correlation between sensors for all engines in the dataset.
    ======================================

    Input:
        data_df (dataframe) - Dataframe with engine data to be plotted.

    Output:
        None.
    """

    # Remove the time column from the dataframe.
    modified_df = data_df.drop('Cycles', axis = 1)

    # Create correlation.
    sensor_corr = modified_df.corr()

    # Define and show correlation plot.
    corr_fig = sns.heatmap(sensor_corr, xticklabels = sensor_corr.columns.values, yticklabels = sensor_corr.columns.values, cmap="YlGnBu")
    plt.title('Engine Data Correlation')
    
    # Save the plot.
    fig_name = r'\engine_data_correlation.png'
    save_string = plot_storage_string + fig_name

    figure = corr_fig.get_figure()
    figure.savefig(save_string, format = 'png', dpi = 600)
