'''
NASA Turbofan Failure Prediction - PyTorch Neural Network Model

This file supports training and evalaution of a Neural Network Model, using PyTorch and Tensors. 

'''

###################################
# Module Importations (A - Z Format)
###################################
import torch

import src.ntfp_dataset_preprocessing as dataset_preprocessing

def create_pytorch_tensors(data_df):
    """
    Create and return PyTorch Tensors from input dataframe.
    ======================================

    Args:
        data_df (dataframe) - Dataframe with engine data to be modelled.

    Returns:
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor (Tensors) - PyTorch Tensors for use with NN models.
    """

    # Create numpy arrays of training and test data using helper method. 
    X_train_array, X_test_array, y_train_array, y_test_array = dataset_preprocessing.prepare_training_data(data_df, 'Cycles')

    # Convert each array into Tensor.
    X_train_tensor = torch.from_numpy(X_train_array)
    X_test_tensor = torch.from_numpy(X_test_array)
    y_train_tensor = torch.from_numpy(y_train_array)
    y_test_tensor = torch.from_numpy(y_test_array)

    # Return Tensors.
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def create_pytorch_NN(input_dimension):
    """
    Initialise and return PyTorch neural network.
    ======================================

    Args:
        input_dimension (int) - Input dimension used in first layer of model (Use training data shape).

    Returns:
        nn (Neural Network) - PyTorch Neural Network.
    """

    # Define model & network.
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dimension, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 1)
        )

    # Return model.
    return model

def train_pytorch_NN():
    """
    Train PyTorch neural network using Tensors.
    ======================================

    Args:
        tensors (Tensors) - PyTorch tensor training data.
        model (Neural Network) - PyTorch neural network, untrained.
        iterations (Int) - Number of training iterations to perform.

    Returns:
        trained_model (Neural Network) - PyTorch Neural Network model, trained.
    """

    # Initialise Loss Function, optimiser and learning rate.

    # Train the model over iteration cycles.

        # Forward pass: Compute predicted y from model working on x.

        # Compute and print loss.

        # Zero gradients, performa a backward pass, and update weights.

    # Return weight-adjusted model.

    pass

def evaluate_pytorch_NN():
    """
    Evaluate PyTorch neural network.
    ======================================

    Args:
        nn (Neural Network) - PyTorch neural network to undergo evaluation.
        test_data (Tensors) - PyTorch Tensors on which model is evaluated.

    Returns:
        evaluation_results (tuple) - Results of Neural Network evaluation.
    """

    pass

def build_train_evaluate_pytorch_NN(data_df):
    """
    Build, train and evaluate (master method) PyTorch neural network.
    ======================================

    Args:
        data_df (dataframe) - Dataframe with engine data to be modelled.

    Returns:
        None.
    """

    pytorch_df = data_df

    # Create training and test tensors.
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = create_pytorch_tensors(pytorch_df)

    # Create model.
    pytorch_model = create_pytorch_NN(X_train_tensor.shape[1])

    # Compile model.
    learning_rate = 1e-3
    opt = torch.optim.Adam(pytorch_model.parameters(), lr = learning_rate)
    pytorch_model.

    # Train model.

    # Evaluate model.

    pass