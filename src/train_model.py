"""
the train_model.py script a neural network to recognize ten different hand gestures.
The trained model will be saved to a file named gesture_recognition_model.h5.
The script assumes that the training and test data are saved in HDF5 files named training_data.h5 and test_data.h5.
"""

# Import the necessary libraries
import model

# Print the docstring
print(__doc__)

# Define the paths to the training and test HDF5 files
training_h5_path = '../data/train/training_data.h5'
test_h5_path = '../data/test/test_data.h5'

# Define the hyperparameter search space
param_distributions = {
    "batch_size": [31, 32],
    "epochs": [2, 3],
}

# Prepare the data
x_train, y_train, x_test, y_test = model.prepare_data(training_h5_path, test_h5_path)

# Train and evaluate the model
model.train_and_evaluate_model(x_train, y_train, x_test, y_test, param_distributions, n_iter=1, cv=3, n_jobs=-1, verbose=1)
