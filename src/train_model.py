"""
the train_model.py script a neural network to recognize ten different hand gestures.
The trained model will be saved to a file named gesture_recognition_model.h5.
The script assumes that the training and test data are saved in HDF5 files named training_data.h5 and test_data.h5.
"""

# Import the necessary libraries
import model
import tensorflow as tf

# Print the docstring
print(__doc__)

# Define the paths to the training and test HDF5 files
training_h5_path = '../data/train/training_data.h5'
test_h5_path = '../data/test/test_data.h5'

# Define the hyperparameter search space
batch_size = [16, 32, 64, 128, 256, 512]
epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
hidden_layers = [1, 2, 3]
neurons_layer1 = [16, 32, 64, 128, 256]
neurons_layer2 = [16, 32, 64, 128, 256]
neurons_layer3 = [16, 32, 64, 128, 256]
batch_normalization = [True, False]
dropout = [True, False]
param_distributions = dict(
    batch_size=batch_size,
    epochs=epochs,
    model__hidden_layers=hidden_layers,
    model__neurons_layer1=neurons_layer1,
    model__neurons_layer2=neurons_layer2,
    model__neurons_layer3=neurons_layer3,
    model__batch_normalization=batch_normalization,
    model__dropout=dropout
)

# Prepare the data
x_train, y_train, x_test, y_test = model.prepare_data(training_h5_path, test_h5_path)

# Train and evaluate the model
model.train_and_evaluate_model(x_train, y_train, x_test, y_test, param_distributions, n_iter=10, cv=3, n_jobs=-1, verbose=1)
