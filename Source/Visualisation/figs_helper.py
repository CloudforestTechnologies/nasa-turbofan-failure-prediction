'''
figs_helper

Helper file for figures and other visualisations.

'''

# Module Importations
from datetime import datetime
from Source import constants

# Constants
TITLE_FONTSIZE = 18
SAVE_FORMAT = 'png'
SAVE_DPI = 600

# Helper Method for saving figures
def generate_fig_save_string(filename):
    """Save Figures
    ======================================
    Saves figures using prescribed filename, date/timestamp and directory.
    
    Args:
        filename (string) - Filename to be used for figure.
        
    Returns:
        save_string (string) - String to use for saving file.
    """

    # Save files to project
    filedirectory = constants.return_fig_save_path()
    proj_code = constants.return_project_code()

    # Mint timestamp
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime('%Y_%m_%d-%H_%M_%S')

    # build filepath
    filepath = filedirectory + '/' + proj_code + '_' + filename + '_' + timestamp_str + '.' + SAVE_FORMAT

    # Return save string
    return filepath