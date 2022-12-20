"""
The model.py script provides central functionalities for using the hand gesture recognition model.
"""

# Import the necessary libraries
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
import h5py
import numpy as np
import pandas as pd
from skopt import BayesSearchCV

# Print the docstring
print(__doc__)

# Load the data from a HDF5 file
def load_data(h5_path):

    #print('Start loading data from HDF5 file: ', h5_path)
    # Open the HDF5 file
    h5_file = h5py.File(h5_path, "r")

    # Initialize the data and labels
    x = np.empty((0, 21, 3))
    y = []

    # Load the data and labels from the HDF5 file
    for dataset_name in h5_file['hand_landmarks_group']:

        # Print the dataset name, label, and shape
        #print('Dataset name: ', dataset_name)
        #print('Dataset label:', h5_file['hand_landmarks_group'][dataset_name].attrs['label'])
        #print('Dataset shape:', h5_file['hand_landmarks_group'][dataset_name].shape)

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
    print('X shape:', x.shape)
    print('y shape:', y.shape)
    # Close the HDF5 file
    h5_file.close()
    print('Data loaded from HDF5 file: ', h5_path)

    return x, y

# Preprocess the coordinates
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

# Load the model
def load_model(model_path):

    # Load the model from the HDF5 file
    model = tf.keras.models.load_model(model_path)
    print('Model loaded from HDF5 file: ', model_path)
    # Print the model's architecture
    model.summary()

    return model  

# Save the model
def save_model(model, df):

    # Generate unique 8 digts id based on model config
    model_id = abs(hash(str(model.get_config()))) % (10 ** 8)
    # Save the results to a markdown file
    df.to_markdown("../models/results/cv_results_{}.md".format(model_id))
    print('Results saved to markdown file: ', "../models/results/results_{}.md".format(model_id))
    # Save the model to the HDF5 file
    model.save("../models/model_{}.h5".format(model_id))
    print('Model saved to HDF5 file: ', "../models/model_{}.h5".format(model_id))

# Create the model
def create_model(hidden_layers, neurons_layer1, neurons_layer2, neurons_layer3):

    # Define the model architecture based on the number of hidden layers
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(63,)))
    if hidden_layers == 1:
        model.add(tf.keras.layers.Dense(neurons_layer1, activation="relu"))
        if hidden_layers == 2:
            model.add(tf.keras.layers.Dense(neurons_layer2, activation="relu"))
            if hidden_layers == 3:
                model.add(tf.keras.layers.Dense(neurons_layer3, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    # Compile the model with the Adam optimizer and the categorical cross-entropy loss
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

# Train and evaluate the model
def train_and_evaluate_model(x_train, y_train, x_test, y_test, param_distributions, n_iter, cv, n_jobs, verbose):

    # Set the random seed for reproducibility
    tf.random.set_seed(42)

    model = KerasClassifier(model=create_model, verbose=1)

    # Define the Bayesian optimization search object
    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=param_distributions,
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

    # Save the cv results to a pandas dataframe
    df = pd.DataFrame(bayes_search.cv_results_)

    # Create the model with the best hyperparameters
    best_model = create_model(bayes_search.best_params_["model__hidden_layers"], bayes_search.best_params_["model__neurons_layer1"], bayes_search.best_params_["model__neurons_layer2"], bayes_search.best_params_["model__neurons_layer3"])
    # Fit the model to the training data
    best_model.fit(x_train, y_train, batch_size=bayes_search.best_params_["batch_size"], epochs=bayes_search.best_params_["epochs"], verbose=1)
    # Evaluate the model on the test data
    _, test_acc = best_model.evaluate(x_test, y_test, verbose=1)

    # Print the model summanry, best hyperparameters, train, validation and test accuracy
    best_model.summary()
    print("Best hyperparameters:", bayes_search.best_params_)
    print("Train accuracy:", bayes_search.cv_results_["mean_train_score"][bayes_search.best_index_])
    print("Validation accuracy:", bayes_search.best_score_)
    print('Test accuracy:', test_acc)

    # Save the model
    save_model(best_model, df)
           
# Predict the gesture
def predict_gesture(model, hand_landmarks):
    
    # Convert the landmarks to a NumPy array
    coordinates = np.array([[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark])
    # Preprocess the coordinates
    flattened_coordinates = preprocess_coordinates(coordinates)
    # Predict the gesture
    prediction = model.predict(flattened_coordinates.reshape(1, -1))
    # Get the index of the predicted gesture
    predicted_gesture = np.argmax(prediction)
    # Get the confidence of the prediction
    confidence = prediction[0][predicted_gesture]
    # Print the gesture and confidence
    print("Predicted gesture: {}, confidence: {}".format(predicted_gesture, confidence))
