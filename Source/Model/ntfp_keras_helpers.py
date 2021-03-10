'''
keras_helpers

Helper routines for building & training neural networks using keras API

'''

# Module Importations
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#import keras
import time
from Source import constants

print('[keras_helpers]Tensorflow version:', tf.__version__)
print('[keras_helpers]keras version =', keras.__version__)

def build_multilayer_perceptron(n_hidden = 2, n_neurons = 6, learning_rate = 1e-3, input_shape = [6]):
    """
    Build and compile multilayer perceptron model.
    ======================================

    Input:
        n_hidden (int) = Number of hidden layers in model.
        n_neurons (int) = Number of neurons in hidden layer.
        learning_rate (float) = Learning rate for optimiser.
        input_shape (array) - An array used to set the shape of the layers.
        output_shape (array) - An array used to set the shape of the output.

    Output:
        model (Sequential) - A multilayer perceptron model designed for the data profile.
    """

    # Print input values
    print("Building Model ...")
    print("Hidden Layers: {}, Neurons: {}, LR: {}".format(n_hidden, n_neurons, learning_rate))

    model = Sequential()

    # Create input layer
    model.add(InputLayer(input_shape = input_shape))

    # Add further layers
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, input_shape = input_shape, activation = "relu"))

    # Add output layer
    model.add(Dense(1))

    # Compile model
    #optimiser = Adam(lr = learning_rate)
    optimiser = RMSprop(lr = learning_rate)
    model.compile(loss = "mse", optimizer = optimiser)

    return model

def wrap_model():
    """
    Build and wrap keras-TF model for use with scikit-learn API.
    ======================================

    Input:
        None

    Output:
        wrapped_model (KerasRegressor) - Wrapped model for use with scikit-learn API.
    """

    # Build and wrap model.
    wrapped_model = KerasRegressor(build_multilayer_perceptron)

    # Return model
    return wrapped_model

def name_model(model_type):
    """
    Returns a time-stamped model name.
    ======================================

    Input:
        model_type (string) - Styling or identifier for model.

    Output:
        name (string) - Model name.
    """
    
    # Generate filename
    timestamp = time.strftime('%Y_%m_%d-%H_%M_%S')
    file_format = '.h5'
    project_code = constants.return_project_code()

    model_name = project_code + '_' + model_type + '_' + timestamp + file_format

    return model_name

def make_save_string(file_name):
    """
    Returns a complete save string for saving a model.
    ======================================

    Input:
        file_name (string) - Filename used in save string.

    Output:
        save_string (string) - Save string for specified model.
    """
    
    # Filepath for saved file
    filepath = constants.return_model_save_path()

    filepath_full = filepath + '\\' + file_name

    return filepath_full

def save_model(model, name):
    """
    Save model to Models folder using described name.
    ======================================

    Input:
        model (Sequential) - The model to be saved.
        name (string) - Model unique identifier

    Output:
        None.
    """

    # Create full filepath, including name
    filename = name_model(name)
    filepath_full = make_save_string(filename)
    print("Save Path:", filepath_full)

    # Save model
    model.save(filepath = filepath_full, overwrite = True, include_optimizer = True)