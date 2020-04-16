'''
NASA Turbofan Failure Prediction - Exploratory Data Analysis

This file supports exploratory data analysis on datasets.

'''

###################################
# Module Importations (A - Z Format)
###################################
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Constants
plot_storage_string = r'C:\Users\ASUS-PC\OneDrive\Cloudforest Technologies\M. Projects\Pink Moss\WP2.0 - NASA Turbofan Failure Prediction\Images'

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


def plot_time_history_all_engines(dataset_df):
    """
    Plot and save the complete engine time history (normalised to RUL) for all engines in dataset.
    ======================================

    Input:
        dataset_df (dataframe) - Dataframe with engine data to be plotted.

    Output:
        None.
    """

    # Prepare dataset for plotting.

    # Define and show correlation plot.

    # Save the plot. 

    pass