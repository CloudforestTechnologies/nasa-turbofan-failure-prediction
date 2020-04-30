'''
NASA Turbofan Failure Prediction - Data Preprocessing

This file supports performing preprocessing operations on datasets.

'''

###################################
# Module Importations (A - Z Format)
###################################
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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

    # Remove columns.
    processed_dataset = dataset.drop(columns = columns_to_be_removed, axis = 1)

    return processed_dataset

def find_correlated_data(data_df, correlation_threshold):
    """
    Identifies column values with strong correlation to other column values.
    ======================================

    Input:
	    data_df (dataframe) - Dataframe containing data to be analysed.
        correlation_threshold (float) - Thershold value above which correlation is assumed.

    Output:
	    data_with_correlation (tuple) - Tuple of two-column data sets, with correlation.
    """

    # Remove time column.
    modified_df = data_df.drop('Cycles', axis = 1)
    
    # Compute pairwise correlation of columns.
    data_corr = modified_df.corr()
    
    # Analyse data correlations; Keep values above threshold.
    high_correlation = []

    for column_no, column in enumerate(data_corr):
        # Create slice to ignore symmetry duplicates.
        col_corr = data_corr[column].iloc[column_no + 1:]

        # Use a bool mask to identify highly-correlated data.
        mask_pairs = col_corr.apply(lambda x: abs(x)) > correlation_threshold

        index_pairs = col_corr[mask_pairs].index

        # Create list of highly-correlated data sets.  
        for index, correlation in zip(index_pairs, col_corr[mask_pairs].values):
            high_correlation.append((column, index, correlation))

    for correlation in high_correlation:
        print(correlation)

    return high_correlation

def list_correlated_data(correlated_data):
    """
    Creates a list of data entities from correlated data tuple.
    ======================================

    Input:
	    correlated_data (tuple) - Tuple of data columns with high correlation.

    Output:
	    data_entities (list) - List of data columns correlated at least once.
    """

    data_list = []

    # Iterate over correlated data, add second value to list if not present.
    for correlation in correlated_data:

        # Data item to be removed is 2nd item in tuple.
        data_item = correlation[1]

        if data_list.__contains__(data_item):
            pass

        else:
            data_list.append(data_item)

    # Return list.
    return data_list

def find_time_independent_columns(data_df):
    """
    Returns a list of columns that do not change with time.
    ======================================

    Input:
	    data_df (dataframe) - Dataframe containing time-series data.
        
    Output:
	    unchanging_columns (list) - List of columns from dataframe which do not change with time.
    """
    
    unchanging_columns = []
    std_threshold = 0.0001

    # Iterate over columns; Identify if std is close-coupled to mean.
    for column in data_df.columns:

        if (data_df[column].std() <= std_threshold * data_df[column].mean()):

            if unchanging_columns.__contains__(column):
                pass

            # Add tightly-coupled columns to list.
            else:
                unchanging_columns.append(column)
  
    return unchanging_columns

def add_calculated_rul(dataset_df):
    """
    Calculates Remaining Useful Life (RUL) and adds it to dataset, which is returned.
    ======================================

    Input:
	    dataset_df (dataframe) - Dataframe containing time-series data.
        
    Output:
	    rul_dataset_df (dataframe) - Returned dataset, inclusive of calculated RUL values.
    """

    rul_dataset_df = dataset_df

    # Iterate over each engine, calculating & adding engine remaining life.
    for engine in rul_dataset_df.index.unique():

        print("Calculating & Appending RUl for Engine " + str(engine))

        # RUL is negative and trends to zero (end of life point)
        #rul_dataset_df.loc[engine, 'RUL'] = rul_dataset_df.loc[engine]['Cycles'].apply(lambda x: x - rul_dataset_df.loc[engine]['Cycles'].max())
        rul_dataset_df.loc[engine, 'RUL'] = rul_dataset_df.loc[engine]['Cycles'].apply(lambda x: rul_dataset_df.loc[engine]['Cycles'].max() - x)

    return rul_dataset_df

def prepare_training_data(dataset_df, target_value, apply_pca = False):
    """
    Prepare and return training and test data arrays from input dataframe, normalising using 
    ======================================

    Input:
        dataset_df (dataframe) - Dataframe containing training dataset.
        target_value (str) - Target value for model training.
        apply_pca (bool) - Determines whether to apply Principle Component Analysis to data.

    Output:
        X_train, X_test, y_train, y_test = Arrays containing split data for model training.
    """

    # Normalisation of training data.

    X_dataset = dataset_df.drop(target_value, axis = 1)

    scalar = StandardScaler()
    # scalar = MinMaxScalar()
    X_array = scalar.fit_transform(X_dataset)

    # Apply PCA, if applicable.
    if (apply_pca == True):

        pca = PCA(n_components = 3)
        X_array = pca.fit_transform(X_array)

    y_array = dataset_df[target_value].values

    # Create split between training and test sets.
    print("Preparing Training Data ...")
    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size = 0.2, random_state = 0)

    return X_train, X_test, y_train, y_test

def standardise_columns(dataset_df):  
    """
    Normalises and returns an [array] of data from an input dataset.
    ======================================

    Input:
	    dataset_df (dataframe) - Dataframe containing un-normalised data.
        
    Output:
	    normalised_data (array) - Normalised data, returned as an array.
    """
    
    initial_dataset = dataset_df

    # Create an array of raw data.
    data_columns = initial_dataset.columns.values[1:-1]
    initial_data = initial_dataset[data_columns].values

    # Transform the data with a scalar method.
    standard_scale = StandardScaler()
    normalised_data = standard_scale.fit_transform(initial_data)

    # Return the transformed data.
    return normalised_data

def calculate_slopes_all_engines(dataset_df, normalised_array):
    """
    Calculate and return a dataframe with data slopes for all columns.
    ======================================

    Input:
	    dataset_df (dataframe) - Dataframe containing un-normalised data.
        normalised_array (array) - Array containing normalised data.
        
    Output:
	    data_slopes_df (dataframe) - Dataframe containing slope data for each original data column.
        slopes_array (array) - Array containing slope data arrayed by engine number.
    """
    # Make list of unique engines.
    engines = dataset_df.index.unique().values

    # Initialise empty slopes array.
    slopes_array = np.empty((normalised_array.shape[1], len(engines)))

    # Iterate over each engine and populate slopes array.
    for iterator, engine in enumerate(engines):
        slopes_array[:,iterator] = calculate_data_lin_regr(dataset_df, normalised_array, engine)

    # Convert slopes to dataframe.
    original_columns = dataset_df.columns.values[1:-1]
    slopes_df = pd.DataFrame(slopes_array.T, index = engines, columns = original_columns)

    # Return dataframe.
    return slopes_df, slopes_array

def calculate_data_lin_regr(dataset_df, normalised_array, engine_number):
    """
    Calculates and returns slopes of linear regression lines for each data column.
    ======================================

    Input:
	    dataset_df (dataframe) - Dataframe containing un-normalised data.
        normalised_array (array) - Array of normalised data.
        engine_number (int) - Engine number for which data is evaluated.
        
    Output:
	    slope (dataframe) - Slope calculations for each data column.
    """

    # Initialise linear regression model.
    model = LinearRegression()

    # Prepare x.
    x = dataset_df.loc[engine_number, 'RUL'].values
    x_rehsaped = x.reshape(-1, 1)

    # Row slice to align with numpy array index.
    row_name = dataset_df.loc[engine_number].iloc[-1].name
    row_slice = dataset_df.index.get_loc(row_name)

    # Prepare y (data values for specific engine).
    y = normalised_array[row_slice]

    # Fit the model.
    model.fit(x_rehsaped, y)

    # Retrieve and return line slope from model.
    slopes = model.coef_[:, 0]
    return slopes

def return_data_ordered_abs_value(slopes_array, dataset_df):
    """
    Orders, prints and returns a list of data values sorted by absolute value.
    ======================================

    Input:
        slopes_array (array) - Array containing normalised column data.
        dataset_df (dataframe) - Dataframe containing parent dataset.
        
    Output:
	    slope_order (array) - An array of data columns, ordered by abs value (highest first).
    """

    # Order slope data as an array.
    slope_order_array = np.argsort(np.abs(slopes_array.mean(axis = 1)))[::-1]

    # Determine columns names as an array.
    data_columns = dataset_df.columns.values[1:-1]

    # Print columns in order of absolute value.
    print('Slope Order: \n{}'.format(data_columns[slope_order_array]))

    # Return slope order data array.
    return slope_order_array