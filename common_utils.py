
#define the root of the data from https://github.com/deep-learning-with-pytorch/dlwpt-code
DATA_ROOT = 'e:/github/dlwpt-type/data'

# we don't git submit the data in this folder
DOWNLOAD_DATA_ROOT = 'e:/github/dlwpt-type/downloaded_data'

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

def inspect_tensors(ts):
    print('=' * 30)
    for t in ts:
        print('-' * 10)
        print(f"tensor:{t}")
        print(f"stride:{t.stride()}")
        print(f"data_ptr:{t.data_ptr()}")
        print(f"untyped_storage().data_ptr:{t.untyped_storage().data_ptr()}")