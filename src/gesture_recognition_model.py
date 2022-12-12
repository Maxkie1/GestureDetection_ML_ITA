"""
Trains a neural network to recognize gestures from images of hands.
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
    # Open the HDF5 file
    h5_file = h5py.File(h5_path, "r")

    # Initialize the data and labels
    x = np.empty((0, 42))
    y = []

    # Load the data and labels from the HDF5 file
    for dataset_name in h5_file['hand_landmarks_group']:

        # Print the dataset name, label, and shape
        print('Dataset loaded: ', dataset_name)
        print('Dataset label:', h5_file['hand_landmarks_group'][dataset_name].attrs['label'])
        print('Dataset shape:', h5_file['hand_landmarks_group'][dataset_name].shape)

        # Append the data to the list of data
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

    # Return the data and labels
    return x, y

# Define the paths to the training and test HDF5 files
training_h5_path = '../data/train/training_data.h5'
test_h5_path = '../data/test/test_data.h5'
# Load the data
x_train, y_train = load_data(training_h5_path)
x_test, y_test = load_data(test_h5_path)

# Shift the labels so that they start at 0
y_train -= 1
y_test -= 1
# Shuffle training data and labels
indices = np.arange(x_train.shape[0])
np.random.seed(42)
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]
# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
# Print the shape of the one-hot labels
print('one-hot y_train shape:', y_train.shape)
print('one-hot y_test shape:', y_test.shape)

# Define the hyperparameter search space
param_distributions = {
    "batch_size": [16, 32, 64, 128],
    "epochs": [20, 40, 60, 80, 100],
}

# Define the model's architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(42,)),
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
    n_iter=1,
    scoring="accuracy",
    cv=3,
    random_state=42,
    n_jobs=-1,
    return_train_score=True,
    verbose=1,
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
