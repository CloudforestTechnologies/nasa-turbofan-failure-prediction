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
plot_storage_string = r'C:\Users\paulm\OneDrive\Cloudforest Technologies\M. Projects\Pink Moss\WP2.0 - NASA Turbofan Failure Prediction\Images'

def plot_time_history_all_engines(dataset_df):
    """
    Plot and save the complete engine time history (normalised to RUL) for all engines in dataset.
    ======================================

    Args:
        dataset_df (dataframe) - Dataframe with engine data to be plotted.

    Returns:
        None.
    """

    # Prepare dataset for plotting.
    plotted_dataset_df = dataset_df.copy()

    plotted_dataset_df.drop('Cycles', axis = 1, inplace = True)

    columns = plotted_dataset_df.columns

    # Define and show plot.
    fig, axes = plt.subplots(len(columns) - 1, 1, figsize = (19, 17))

    for column, ax in zip(columns, axes):

        print("Plotting Engine RUL Data: " + str(column))

        # Filter for the RUL data only.
        if column == 'RUL':
            continue

        fontdict = {'fontsize': 14}
        ax.set_title(column, loc = 'left', fontdict = fontdict)

        # Add data for each engine to axis.
        for engine in plotted_dataset_df.index.unique():
            rul_time = plotted_dataset_df.loc[engine, 'RUL']
            ax.plot(rul_time, plotted_dataset_df.loc[engine, column], label = column)

    # Add figure title.
    fig.suptitle('Run to Failure - All Engines')

    # Save the plot.
    fig_name = r'\time_history_all_engines.png'

    save_string = plot_storage_string + fig_name

    fig.savefig(save_string, format = 'png', dpi = 600)