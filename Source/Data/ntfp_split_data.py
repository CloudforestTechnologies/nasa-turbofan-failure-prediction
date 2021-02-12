'''
ntfp_split_data

This file supports splitting data into development and validation sets, ensuring randomisation of the sets and the same distribution on each occurence.
'''

# Module Importations
import numpy as np
import pandas as pd

def split_train_eval(data, split_ratio):
    """Split Training & Evaluation Data
    ======================================
    Splits original dataset into training and evaluation data.
    
    Args:
        data (dataframe) - Original test data.
        split_ratio (int) - Ratio for splitting dataset as evaluation fraction.
        
    Returns:
        data_train (dataframe) - Dataframe with training data slice.
        data_eval (dataframe) - Dataframe with evaluation data slice.
    """

    # Random seed setting ensures identical data split between calls
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))

    train_set_size = int(len(data) * split_ratio)

    # Create slices of training and evaluation indices
    train_indices = shuffled_indices[train_set_size:]
    eval_indices = shuffled_indices[:train_set_size]

    # Create training and evaluation datasets
    training_data = data.iloc[train_indices]
    evaluation_data = data.iloc[eval_indices]

    # Check length
    print("Original Data Items:", len(data))
    print("Training Data Items:", len(training_data))
    print("Evaluation Data Items:", len(evaluation_data))

    assert len(data) == (len(training_data) + len(evaluation_data))

    # Return datasets
    return training_data, evaluation_data
