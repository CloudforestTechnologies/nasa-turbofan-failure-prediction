'''
TensorBoard_Helpers

Helper routines for visualising model training performance using TensorBoard.

'''

# Module Importations
import os
import time

# Custom Module Imports
from Source import constants

def get_run_logdir():
    """
    Return the log directory for a new run.
    ======================================

    Input:
        None.

    Output:
        logdir (String) - Log directory with time stamp.
    """

    # Mint time stamp
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")

    # Create path
    log_path = constants.return_tensorboard_log_path() 

    # Return directory
    logdir = os.path.join(log_path, run_id)

    return logdir