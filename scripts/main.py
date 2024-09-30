import subprocess

command = [
    'python3',
    '../training.py', 
    '../training_testing_data/train/LR', 
    '../training_testing_data/train/HR', 
    '../training_testing_data/test/LR', 
    '../training_testing_data/HR'
]

# Run the command
result = subprocess.run(command, capture_output=False, text=True)

