# TODO: convert all the TIFF files in the OLI2MSI to .npy files
# and store them in the HR and LR directories

import tifffile as tf
import numpy as np
import os

def convert_to_npy(path: str):
    # Convert TIFF file to npy array
    array = tf.imread(path)
    # Change 3 channel array to 2d array of just red channel
    array = array[0, :, :]
    return array

def get_type(root: str):
    # returns a tuple (train/test, hr/lr)
    if "test_hr" in root:
        return 'test', 'HR'
    if "test_lr" in root:
        return 'test', 'LR'
    if "train_hr" in root:
        return 'train', 'HR'
    if "train_lr" in root:
        return 'train', 'LR'

if __name__ == "__main__":
    
    main_directory = "OLI2MSI"

    i = 0
    for root, dirs, files in os.walk(main_directory):
        for file in files:
            i += 1
            file_path = os.path.join(root, file)
            file_type = get_type(root) # Returns the type of the file 

            array = convert_to_npy(f'{root}/{file}')
            
            # Save file as .npy
            np.save(f'{file_type[0]}/{file_type[1]}/img{i}.npy', array)

