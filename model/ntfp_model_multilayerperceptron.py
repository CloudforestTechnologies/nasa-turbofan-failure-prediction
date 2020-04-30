'''
NASA Turbofan Failure Prediction - Multi-Layer Peceptron

This file supports training and evaluating a multi-layer perceptron regression model, using Keras. 

'''

###################################
# Module Importations (A - Z Format)
###################################
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import keras
import numpy as np

def prepare_training_data(dataset_df, target_value):
    """
    Prepare and return training and test data arrays from input dataframe.
    ======================================

    Input:
        dataset_df (dataframe) - Dataframe containing training dataset.
        target_value (str) - Target value for model training.

    Output:
        X_train, X_test, y_train, y_test = Arrays containing split data for model training.
    """

    # Additional scaling of data for neural network.

    X_dataset = dataset_df.drop(target_value, axis = 1)

    scalar = MinMaxScaler()
    X_array = scalar.fit_transform(X_dataset)

    y_max = dataset_df[target_value].max()
    y_mean = dataset_df[target_value].mean()
    y_array = (dataset_df[target_value].values) / y_max

    # Create split between training and test sets.
    print(print("[mlp Neural Network] Preparing Training Data ..."))
    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size = 0.2, random_state = 0)

    return X_train, X_test, y_train, y_test

def create_multilayer_perceptron(dim):
    """
    Build and return a multi-layer perceptron model using Keras.
    ======================================

    Input:
        dim (array) - An array used to set the shape of the layers.

    Output:
        model (Sequential) - A multi-layer perceptron model designed for the data profile.
    """

    # Define the network.
    model = Sequential()
    model.add(Dense(12, input_dim = dim, activation = "relu"))
    model.add(Dense(8, activation = "relu"))
    model.add(Dense(4, activation = "relu"))
    model.add(Dense(1, activation = "linear"))

    return model

def train_multi_layer_NN_model(dataset_df, target_value):
    """
    Creates, trains and returns a neural network model from training data.
    ======================================

    Input:
        dataset_df (dataframe) - Dataframe containing training dataset.
        target_value (str) - Target value for model training.

    Output:
        mlp_model (Sequential) - Multi-layer perceptron model, fitted to training data.
    """

    # Clear existing models.
    keras.backend.clear_session()

    # Prepare training and test datasets.
    X_train, X_test, y_train, y_test = prepare_training_data(dataset_df, target_value)

    # Create the model.
    mlp_model = create_multilayer_perceptron(X_train.shape[1])

    # Compile the model using mean absolute percentage error as loss.
    opt = Adam(lr = 1e-3, decay = 1e-3 /200)
    mlp_model.compile(loss = "mean_absolute_percentage_error", optimizer = opt)

    # Train the model.
    print("[mlp Neural Network] Training model ...")
    mlp_model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 50, batch_size = 8)

    # Make predictions on the test data.
    y_pred = mlp_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    print("mlp Neural Network MAE: " + str(mae))

    rmse = mean_squared_error(y_test, y_pred, squared = False)
    print("mlp Neural Network MSE: " + str(rmse))

    return mlp_model