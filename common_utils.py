
#define the root of the data from https://github.com/deep-learning-with-pytorch/dlwpt-code
DATA_ROOT = 'e:/github/dlwpt-type/data'

import os

def get_files_with_extensions(folder_path, extensions):
    """
    Get a list of filenames in a folder with specified extensions.

    Parameters:
    - folder_path (str): The path to the folder.
    - extensions (list): A list of file extensions to filter.

    Returns:
    - list: A list of filenames with specified extensions.
    """
    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(tuple(extensions))]
    return file_list