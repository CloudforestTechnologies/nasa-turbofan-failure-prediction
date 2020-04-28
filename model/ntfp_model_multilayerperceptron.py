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
import numpy as np

def prepare_training_data(dataset_df, target_value):
    """
    Prepare and return training and test data arrays from input dataframe.
    ======================================

    Input:
        dataset_df (dataframe) - Dataframe containing training dataset.
        target_value (str) - Target value for model training.

    Output:
        X_train, X_test, y_train, y_test = Arrays containing data for model training.
    """
    pass

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
    model.add(Dense(8, input_dim = dim, activation = "relu"))
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

    # Prepare training and test datasets.

    # Create the model.
    model = create_multilayer_perceptron(trainX.shape[1], regress = True)

    # Compile the model using mean absolute percentage error as loss.
    opt = Adam(lr = 1e-3, decay = 1e-3 /200)
    model.compile(loss = "mean_absolute_percentage_error", optimizer = opt)

    # Train the model.
    print("[mlp Neural Network] Training model ...")
    model.fit(trainX, trainY, validation_data = (testX, testY), epochs = 200, batch_size = 8)

    # Make predictions on the test data.
    print("[INFO] Predicting house prices ...")
    predictions = model.predict(testX)