'''
NASA Turbofan Failure Prediction - Data Preprocessing

This file supports performing preprocessing operations on datasets.

'''

###################################
# Module Importations (A - Z Format)
###################################

def dataset_remove_columns(dataset, columns_to_be_removed):
    """
    Removes selected columns from dataframe, returns a new data frame.
    ======================================

    Input:
	    dataset (dataframe) - Initial dataframe, from which columns are to be removed.
        columns_to_be_removed (list) - List of column names to be removed from dataframe.

    Output:
	    processed_dataset (dataframe) - Final dataframe, with specified columns removed.
    """

    pass

def find_correlated_data(dataset, correlation_threshold):
    """
    Identifies discrete data with strong correlation to other readings.
    ======================================

    Input:
	    dataset (dataframe) - Dataframe containing data to be analysed.
        correlation_threshold (float) - Thershold value above which correlation is assumed.

    Output:
	    data_with_correlation (list) - List of two-column data sets with correlation.
    """

    pass