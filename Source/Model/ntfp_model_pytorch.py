'''
NASA Turbofan Failure Prediction - PyTorch Neural Network Model

This file supports training and evalaution of a Neural Network Model, using PyTorch and Tensors. 

'''

###################################
# Module Importations (A - Z Format)
###################################
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
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

    # Convert Tensor data type to float.
    X_train_tensor = X_train_tensor.float()
    X_test_tensor = X_test_tensor.float()
    y_train_tensor = y_train_tensor.float()
    y_test_tensor = y_test_tensor.float()

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

def train_pytorch_NN(X_train, y_train, model, epochs = 200):
    """
    Train PyTorch neural network using Tensors.
    ======================================

    Args:
        X_train (Tensor) - PyTorch tensor training data.
        y_train (Tensor) - PyTorch tensor training data.
        model (Neural Network) - PyTorch neural network, untrained.
        epochs (Int) - Number of training epochs to perform.

    Returns:
        model (Neural Network) - PyTorch Neural Network model, trained.
    """

    # Reshape y_train to work with model output / loss.
    y_train = y_train.view(-1, 1)

    # Initialise Loss Function, optimiser and learning rate.
    loss_func = torch.nn.MSELoss()
    learning_rate = 1e-3
    opt = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # Train the model over iteration cycles.
    for epoch in range(epochs):

        # Forward pass: Compute predicted y from model working on x.
        y_pred = model(X_train)

        # Compute and print loss.
        loss = loss_func(y_pred, y_train)
        print("Epoch: " + epoch.__str__(), loss.item().__str__())

        # Zero gradients, perform a backward pass, and update weights.
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Return weight-adjusted model.
    return model

def evaluate_pytorch_NN(model, X_test, y_test):
    """
    Evaluate PyTorch neural network.
    ======================================

    Args:
        model (Neural Network) - PyTorch neural network to undergo evaluation.
        X_test (Tensor) - PyTorch Tensor on which model is evaluated.
        y_test (Tensor) - PyTorch Tensor on which model is evaluated.

    Returns:
        None.
    """

    # Make predictions.
    y_pred = model(X_test)

    # Reshape y_test, if needed.
    y_test = y_test.view(-1, 1)

    # Convert to numpy array if needed.
    y_test = y_test.numpy()
    y_pred = y_pred.detach().numpy()

    # Evaluate predictions.
    mae = mean_absolute_error(y_test, y_pred)
    print("PyTorch NN MAE: " + str(mae))

    rmse = mean_squared_error(y_test, y_pred, squared = False)
    print("PyTorch NN MSE: " + str(rmse))

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

    # Train model.
    trained_model = train_pytorch_NN(X_train_tensor, y_train_tensor, pytorch_model, 200000)

    # Evaluate model.
    evaluate_pytorch_NN(trained_model, X_test_tensor, y_test_tensor)