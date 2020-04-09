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

def find_correlated_data(data_df, correlation_threshold):
    """
    Identifies discrete data with strong correlation to other readings.
    ======================================

    Input:
	    data_df (dataframe) - Dataframe containing data to be analysed.
        correlation_threshold (float) - Thershold value above which correlation is assumed.

    Output:
	    data_with_correlation (list) - List of two-column data sets with correlation.
    """

    # Remove time column.
    modified_df = data_df.drop('Cycles', axis = 1)
    
    # Compute pairwise correlation of columns.
    data_corr = modified_df.corr()
    
    # Analyse data correlations; Keep values above threshold.
    high_correlation = []

    for column_no, column in data_corr:
        # Create slice to ignore symmetry duplicates.

        # Use a bool mask to identify highly-correlated data.
        mask_pairs = data_corr.apply(lambda x: abs(x)) > correlation_threshold

        index_pairs = data_corr[mask_pairs].index

        # Create list of highly-correlated data sets.  
        for index, correlation in zip(index_pairs, data_corr[mask_pairs].values):
            high_correlation.append((column, index, correlation))

    print(high_correlation)

    return high_correlation