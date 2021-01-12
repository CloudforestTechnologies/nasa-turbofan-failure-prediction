'''
NASA Turbofan Failure Prediction - Project Pipeline

This is the main script of the project.

This script trains and evaluates a series of ML models to predict failure on turbofan engines using data supplied by NASA.  

'''

###################################
# Module Importations (A - Z Format)
###################################
import pandas as pd

import model.ntfp_model_multilayerperceptron as mlp_nn
import model.ntfp_model_pytorch as pytorch_nn
import model.ntfp_model_randomforest as random_forest
import src.ntfp_dataset_baseline as dataset_baseline
import src.ntfp_dataset_exploratory as dataset_eda
import src.ntfp_dataset_import as dataset_import
import src.ntfp_dataset_preprocessing as dataset_preprocessing

###################################
# Main Datapipeline Script
###################################

if __name__ == '__main__':
    
    # Storage container for model performance.

    # Import & peek data.
    #raw_data_df = dataset_import.import_dataset()

    #dataset_import.peek_dataset(raw_data_df)
    
    # Visualise engine sensor correlation data.
    dataset_eda.visualise_sensor_correlation_all_engine(raw_data_df)

    # Reduce / Eliminate highly-correlated sensors.
    correlation_threshold = 0.95
    
    correlated_data = dataset_preprocessing.find_correlated_data(raw_data_df, correlation_threshold)
    
    columns_to_be_removed = dataset_preprocessing.list_correlated_data(correlated_data)
    
    processed_df = dataset_preprocessing.dataset_remove_columns(raw_data_df, columns_to_be_removed)

    print(processed_df.info)

    # Visualise distribution of sensor values.
    dataset_eda.visualise_sensor_data_distribution(processed_df)

    # Remove data that does not change with time.
    time_independent_columns = dataset_preprocessing.find_time_independent_columns(processed_df)
    
    processed_df = dataset_preprocessing.dataset_remove_columns(processed_df, time_independent_columns)

    print(processed_df.info)

    # Add Remaining Useful Life (RUL) to dataset.
    rul_dataset = dataset_preprocessing.add_calculated_rul(processed_df)

    print(rul_dataset.head())

    # Visualise sensor behaviour against RUL.
    dataset_eda.plot_time_history_all_engines(rul_dataset)

    # Remove data columns with no apparent trend.
    data_columns_no_trend = ['Set-1', 'Set-2']
    rul_dataset = dataset_preprocessing.dataset_remove_columns(rul_dataset, data_columns_no_trend)
    
    print(rul_dataset.head())

    # Standardise remaining data columns.
    normalised_data = dataset_preprocessing.standardise_columns(rul_dataset)

    # Calculate slope for each data value via linear regression.
    slopes_df, slopes_array = dataset_preprocessing.calculate_slopes_all_engines(rul_dataset, normalised_data)
    print(slopes_df.describe())

    # Order slopes by value.
    slope_order = dataset_preprocessing.return_data_ordered_abs_value(slopes_array, rul_dataset)

    # Drop all data columns except [5] most influential, using array operators to identify & slice column names.
    slope_slice = slope_order[5:]
    data_columns = rul_dataset.columns.values[1:-1]
    data_columns_not_influential = data_columns[slope_slice]
    print(data_columns_not_influential)
    rul_dataset = dataset_preprocessing.dataset_remove_columns(rul_dataset, data_columns_not_influential)

    print(rul_dataset)

    # Save processed data set.
    rul_dataset.to_pickle(r'D:\Developer Area\nasa_turbofan_failure_prediction\data\normalised_data.pkl')
    
    rul_dataset = pd.read_pickle(r'D:\Developer Area\nasa_turbofan_failure_prediction\data\normalised_data.pkl')

    # Create baseline ML model tracking against RUL.
    baseline_model = dataset_baseline.create_baseline_model(rul_dataset, 'RUL', apply_pca = False)

    # Train / evaluate random forest regressor.
    random_forest_model = random_forest.train_random_forest_model(rul_dataset, 'RUL', apply_pca = False)

    # Train / evaluate NN.
    mlp_NN_model = mlp_nn.train_multi_layer_NN_model(rul_dataset, 'RUL', apply_pca = False)

    # Train / evaluate PyTorch nn.
    pytorch_nn.build_train_evaluate_pytorch_NN(rul_dataset)
    
    # Retrain and evaluate models with PCA enabled.
    baseline_model_PCA = dataset_baseline.create_baseline_model(rul_dataset, 'RUL', apply_pca = True)

    random_forest_model_PCA = random_forest.train_random_forest_model(rul_dataset, 'RUL', apply_pca = True)

    mlp_NN_model_PCA = mlp_nn.train_multi_layer_NN_model(rul_dataset, 'RUL', apply_pca = True)