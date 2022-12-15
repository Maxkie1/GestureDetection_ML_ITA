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

# Confirm that we can connect Tensorflow to the GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Define the paths to the training and test HDF5 files
training_h5_path = '../data/train/training_data.h5'
test_h5_path = '../data/test/test_data.h5'

# Define the hyperparameter search space
batch_size = [800, 1000]
epochs = [1, 3]
neurons = [16, 35]
param_distributions = dict(
    batch_size=batch_size,
    epochs=epochs,
    model__neurons=neurons,
    
)


# Prepare the data
x_train, y_train, x_test, y_test = model.prepare_data(training_h5_path, test_h5_path)

# Train and evaluate the model
model.train_and_evaluate_model(x_train, y_train, x_test, y_test, param_distributions, n_iter=2, cv=2, n_jobs=-1, verbose=1)
