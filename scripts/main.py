import os
import subprocess

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Build paths relative to the script directory
command = [
    'python3', 
    os.path.join(script_dir, '../training.py'),
    os.path.join(script_dir, '../training_testing_data/train/LR'),
    os.path.join(script_dir, '../training_testing_data/train/HR'),
    os.path.join(script_dir, '../training_testing_data/test/LR'),
    os.path.join(script_dir, '../training_testing_data/HR')
]
# Run the command
result = subprocess.run(command, capture_output=False, text=True)

