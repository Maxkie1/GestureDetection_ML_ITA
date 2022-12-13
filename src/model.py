"""
Trains a neural network to recognize ten different hand gestures.
The trained model will be saved to a file named gesture_recognition_model.h5.
"""

# Import the necessary libraries
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
import h5py
import numpy as np
from skopt import BayesSearchCV

# Print the docstring
print(__doc__)

# Load the data from a HDF5 file
def load_data(h5_path):

    print('Start loading data from HDF5 file: ', h5_path)
    # Open the HDF5 file
    h5_file = h5py.File(h5_path, "r")

    # Initialize the data and labels
    x = np.empty((0, 21, 3))
    y = []

    # Load the data and labels from the HDF5 file
    for dataset_name in h5_file['hand_landmarks_group']:

        # Print the dataset name, label, and shape
        print('Dataset name: ', dataset_name)
        print('Dataset label:', h5_file['hand_landmarks_group'][dataset_name].attrs['label'])
        print('Dataset shape:', h5_file['hand_landmarks_group'][dataset_name].shape)

        # Append the data to the list of x
        x = np.append(x, h5_file['hand_landmarks_group'][dataset_name], axis=0)
  
        # Append the label to the list of labels
        label = h5_file['hand_landmarks_group'][dataset_name].attrs['label']
        for i in range(h5_file['hand_landmarks_group'][dataset_name].shape[0]):
            entry_label = label
            y.append(entry_label)

    # Convert the labels to NumPy arrays
    y = np.array(y)
    # Print the shape of the data and labels
    print('x shape:', x.shape)
    print('y shape:', y.shape)
    # Close the HDF5 file
    h5_file.close()
    print('Data loaded from HDF5 file: ', h5_path)

    return x, y

# Preprocess data
def preprocess_coordinates(coordinates):
    
    # Get the wrist landmark as the origin for the relative coordinates
    wrist = coordinates[0]
    # Calculate the relative coordinates and flip the hand
    relative_coordinates = -(coordinates - wrist)
    # Normalize the relative coordinates
    normalized_coordinates = (relative_coordinates - np.min(relative_coordinates)) / np.ptp(relative_coordinates)
    # Flatten the normalized coordinates
    flattened_coordinates = normalized_coordinates.flatten()

    return flattened_coordinates

# Prepare the data
def prepare_data(training_path, test_path):
    
    # Load the data
    x_train, y_train = load_data(training_path)
    x_test, y_test = load_data(test_path)

    # Preprocess the data
    x_train = np.array([preprocess_coordinates(x) for x in x_train])
    x_test = np.array([preprocess_coordinates(x) for x in x_test])
    print('Training Data preprocessed to shape:', x_train.shape, 'and type:', x_train.dtype, 'and range:', np.min(x_train), '-', np.max(x_train))
    print('Test Data preprocessed to shape:', x_test.shape, 'and type:', x_test.dtype, 'and range:', np.min(x_test), '-', np.max(x_test))

    # Shift the labels so that they start at 0
    y_train -= 1
    y_test -= 1
    print('Labels shifted.')

    # Shuffle training data and labels
    indices = np.arange(x_train.shape[0])
    np.random.seed(42)
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    print('Training data shuffled.')

    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    print('Labels one-hot encoded.')

    return x_train, y_train, x_test, y_test

# Train and evaluate the model
def train_and_evaluate_model(x_train, y_train, x_test, y_test, param_distributions, n_iter, cv, n_jobs, verbose):

    # Define the model's architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    # Print the model's architecture
    model.summary()

    # Compile the model with the Adam optimizer and the categorical cross-entropy loss
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Wrap the model in a KerasClassifier
    wrapped_model = KerasClassifier(model=model, verbose=1)

    # Define the Bayesian optimization search object
    bayes_search = BayesSearchCV(
        wrapped_model,
        param_distributions,
        n_iter=n_iter,
        scoring="accuracy",
        cv=cv,
        random_state=42,
        n_jobs=n_jobs,
        return_train_score=True,
        verbose=verbose,
    )

    # Fit the model to the training data
    bayes_search.fit(x_train, y_train)

    # Print the train, validation, test accuracies and best hyperparameters
    print("Train accuracy:", bayes_search.cv_results_["mean_train_score"][bayes_search.best_index_])
    print("Validation accuracy:", bayes_search.best_score_)
    print("Test accuracy:", bayes_search.score(x_test, y_test))
    print("Best hyperparameters:", bayes_search.best_params_)

    # Save the model to a HDF5 file
    bayes_search.best_estimator_.model.save("../models/gesture_recognition_model.h5")

# Define the paths to the training and test HDF5 files
training_h5_path = '../data/train/training_data.h5'
test_h5_path = '../data/test/test_data.h5'

# Define the hyperparameter search space
param_distributions = {
    "batch_size": [16, 32, 64, 128, 256, 512],
    "epochs": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
}

# Prepare the data
x_train, y_train, x_test, y_test = prepare_data(training_h5_path, test_h5_path)

# Train and evaluate the model
train_and_evaluate_model(x_train, y_train, x_test, y_test, param_distributions, n_iter=1, cv=3, n_jobs=-1, verbose=1)

