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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from matplotlib import pyplot as plt

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

    ## Slice to y_train and y_train to 50000 samples for development purposes
    #x_train = x_train[:50000]
    #y_train = y_train[:50000] 

    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    print('Labels one-hot encoded.')

    return x_train, y_train, x_test, y_test

# Plot the model performance
def plot_model_performance(history, hash):

    # Plot the model performance and loss and save both plots to same file
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label = 'Test')
    plt.xlabel('Epoch')
    plt.ylabel('Model Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label = 'Test')
    plt.xlabel('Epoch')
    plt.ylabel('Model Loss')
    plt.ylim([0, max(plt.ylim())])
    plt.legend(loc='upper right')
    plt.savefig('../models/results/plot_{}.png'.format(hash))

# Plot the model performance visually appealing 
def plot_model_performance_dynamic(history, hash):

    # Plot the model performance and loss and save both plots to same file
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label = 'Test')
    plt.xlabel('Epoch')
    plt.ylabel('Model Accuracy')
    plt.ylim([min(plt.ylim()), max(plt.ylim())])
    plt.legend(loc='lower right')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label = 'Test')
    plt.xlabel('Epoch')
    plt.ylabel('Model Loss')
    plt.ylim([min(plt.ylim()), max(plt.ylim())])
    plt.legend(loc='upper right')
    plt.savefig('../models/results/plot_dynamic_{}.png'.format(hash))

# plot the confusion matrix
def plot_confusion_matrix(y_test, y_pred, model_id):
        
        # Get the confusion matrix
        cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        # Get the relative confusion matrix
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # Round the values to 2 decimals
        cm = np.around(cm, decimals=2)
        # Plot the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
        disp.plot()
        # Save the confusion matrix to a png file
        plt.savefig('../models/results/confusion_matrix_{}.png'.format(model_id))

# Load the model
def load_model(model_path):

    # Load the model from the HDF5 file
    model = tf.keras.models.load_model(model_path)
    print('Model loaded from HDF5 file: ', model_path)
    # Print the model's architecture
    model.summary()

    return model  

# Save the model
def save_model(model, df, history):

    # Generate unique 8 digts id based on model config
    model_id = abs(hash(str(model.get_config()))) % (10 ** 8)
    # Save the results to a markdown file
    df.to_markdown("../models/results/results_{}.md".format(model_id))
    print('Results saved to markdown file: ', "../models/results/results_{}.md".format(model_id))
    # Save the model performance to a png file
    plot_model_performance(history, model_id)
    plot_model_performance_dynamic(history, model_id)
    print('Model performance plot saved to png file: ', "../models/results/plot_{}.png".format(model_id))
    # Save the model to the HDF5 file
    model.save("../models/model_{}.h5".format(model_id))
    print('Model saved to HDF5 file: ', "../models/model_{}.h5".format(model_id))

# Create the model
def create_model(hidden_layers, neurons_layer1, neurons_layer2, neurons_layer3, batch_normalization, dropout):

    # Define the model architecture based on the number of hidden layers
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(63,)))
    if hidden_layers == 1 or hidden_layers == 2 or hidden_layers == 3:
        model.add(tf.keras.layers.Dense(neurons_layer1))
        model.add(tf.keras.layers.Activation('tanh'))
        if batch_normalization == True:
            model.add(tf.keras.layers.BatchNormalization())
        if dropout == True:
            model.add(tf.keras.layers.Dropout(0.2))
        if hidden_layers == 2 or hidden_layers == 3:
            model.add(tf.keras.layers.Dense(neurons_layer2))
            model.add(tf.keras.layers.Activation('tanh'))
            if batch_normalization == True:
                model.add(tf.keras.layers.BatchNormalization())
            if dropout == True:
                model.add(tf.keras.layers.Dropout(0.2))
            if hidden_layers == 3:
                model.add(tf.keras.layers.Dense(neurons_layer3))
                model.add(tf.keras.layers.Activation('tanh'))
                if batch_normalization == True:
                    model.add(tf.keras.layers.BatchNormalization())
                if dropout == True:
                    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('softmax'))

    ## Define the model architecture based on the number of hidden layers
    #model = tf.keras.Sequential()
    #model.add(tf.keras.layers.Input(shape=(63,)))
    #neurons = [neurons_layer1, neurons_layer2, neurons_layer3][:hidden_layers]
    #for i, n in enumerate(neurons):
    #    model.add(tf.keras.layers.Dense(n))
    #    model.add(tf.keras.layers.Activation('relu'))
    #    if batch_normalization:
    #        model.add(tf.keras.layers.BatchNormalization())
    #    if dropout:
    #        model.add(tf.keras.layers.Dropout(0.2))
    #model.add(tf.keras.layers.Dense(10))
    #model.add(tf.keras.layers.Activation('softmax'))

    # Compile the model with the Adam optimizer and the categorical cross-entropy loss
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model

# Train and evaluate the model with bayesian optimization
def train_and_evaluate_model(x_train, y_train, x_test, y_test, param_distributions, n_iter, cv, n_jobs, verbose):

    # Set the random seed for reproducibility
    tf.random.set_seed(42)

    # Create the KerasClassifier model
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

    # Execute the search
    bayes_search.fit(x_train, y_train)

    # Save the cross validation results to a pandas dataframe
    df = pd.DataFrame(bayes_search.cv_results_)

    # Create the model with the best hyperparameters
    best_model = create_model(
        bayes_search.best_params_["model__hidden_layers"], 
        bayes_search.best_params_["model__neurons_layer1"], 
        bayes_search.best_params_["model__neurons_layer2"], 
        bayes_search.best_params_["model__neurons_layer3"], 
        bayes_search.best_params_["model__batch_normalization"], 
        bayes_search.best_params_["model__dropout"])
    # Fit the model to the whole training data (validating on the test data for plotting the learning curves)
    history = best_model.fit(x_train, y_train, batch_size=bayes_search.best_params_["batch_size"], epochs=bayes_search.best_params_["epochs"], validation_data=(x_test, y_test), verbose=1)
    # Evaluate the model on the test data
    _, test_accuracy = best_model.evaluate(x_test, y_test, verbose=1)    

    # Print the model summary, best hyperparameters, train, validation and test accuracy
    best_model.summary()
    print("Best hyperparameters:", bayes_search.best_params_)
    print("Train accuracy:", bayes_search.cv_results_["mean_train_score"][bayes_search.best_index_])
    print("Validation accuracy:", bayes_search.best_score_)
    print("Test accuracy:", test_accuracy)
    
    # Save the model
    save_model(best_model, df, history)

# Train and evaluate a custom model
def train_and_evaluate_custom_model(x_train, y_train, x_test, y_test, param_distributions):
    
    # Set the random seed for reproducibility
    tf.random.set_seed(42)

    # Create the model
    model = create_model(param_distributions["hidden_layers"], param_distributions["neurons_layer1"], param_distributions["neurons_layer2"], param_distributions["neurons_layer3"], param_distributions["batch_normalization"], param_distributions["dropout"])

    # Fit the model to the whole training data (validating on the test data for plotting the learning curves)
    history = model.fit(x_train, y_train, batch_size=param_distributions["batch_size"], epochs=param_distributions["epochs"], validation_data=(x_test, y_test), verbose=1)
    # Evaluate the model on the test data
    _, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    # Predict the labels
    y_pred = model.predict(x_test)

    # Print the model summary, best hyperparameters, train, validation and test accuracy
    model.summary()
    print("Train accuracy:", history.history["accuracy"][-1])
    #print("Validation accuracy:", history.history["val_accuracy"][-1])
    print("Test accuracy:", test_accuracy)
    print("Classification report:", classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
    
    # Save the model
    # Generate unique 8 digts id based on model config
    model_id = abs(hash(str(model.get_config()))) % (10 ** 8)
    # Save the model performance to a png file
    plot_model_performance(history, model_id)
    plot_model_performance_dynamic(history, model_id)
    print('Model performance plot saved to png file: ', "../models/results/plot_{}.png".format(model_id))
    # Save the confusion matrix to a png file
    plot_confusion_matrix(y_test, y_pred, model_id)
    # Save the model architecture to a png file
    tf.keras.utils.plot_model(model, to_file="../models/results/model_architecture_{}.png".format(model_id), show_shapes=True, show_layer_names=True)
    # Save the model to the HDF5 file
    model.save("../models/model_{}.h5".format(model_id))
    print('Model saved to HDF5 file: ', "../models/model_{}.h5".format(model_id))

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

    return predicted_gesture, confidence
