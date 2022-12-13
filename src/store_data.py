"""
The store_data.py script processes the hand landmarks detected in a static image using MediaPipe Hands. 
The landmarks are converted to coordinates which are saved in respective HDF5 files. 
The resulting coordinates can be used as input for the hand gesture recognition model.
"""

# Import the necessary modules
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

# Import the drawing utilities and styles from the MediaPipe Hands module
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Print the docstring
print(__doc__)

# Use the Hands class from the MediaPipe Hands module to process the static images
def process_image_with_hands(file_path):

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        
            # Read an image, flip it around y-axis for correct handedness output
            image = cv2.flip(cv2.imread(file_path), 1)
            # Convert the BGR image to RGB before processing
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # If no hand landmarks are detected, skip to the next iteration
            if not results.multi_hand_landmarks:
                print("No hand landmarks detected: ", file_path)
                return None

            # Iterate over the detected hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:

                # Convert the landmarks to a NumPy array
                coordinates = np.array([[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark])

                return coordinates

# Plot hand landmarks in 2D
def plot_hand_landmarks(coordinates):

    # Plot the normalized coordinates
    fig = plt.figure()
    ax = fig.add_subplot()
    # Plot the points
    ax.scatter(coordinates[:, 0], coordinates[:, 1])
    # Thumb plot
    thumb_coordinates = coordinates.take([0,1,2,3,4], axis=0)
    ax.plot(thumb_coordinates[:, 0], thumb_coordinates[:, 1], color='blue')
    # Index finger plot
    index_finger_coordinates = coordinates.take([0,5,6,7,8], axis=0)
    ax.plot(index_finger_coordinates[:, 0], index_finger_coordinates[:, 1], color='blue')
    # Middle finger plot
    middle_finger_coordinates = coordinates.take([9,10,11,12], axis=0)
    ax.plot(middle_finger_coordinates[:, 0], middle_finger_coordinates[:, 1], color='blue')
    # Ring finger plot
    ring_finger_coordinates = coordinates.take([13,14,15,16], axis=0)
    ax.plot(ring_finger_coordinates[:, 0], ring_finger_coordinates[:, 1], color='blue')
    # Pinky plot
    pinky_coordinates = coordinates.take([0,17,18,19,20], axis=0)
    ax.plot(pinky_coordinates[:, 0], pinky_coordinates[:, 1], color='blue')
    # Finger connection plot
    finger_connection_coordinates = coordinates.take([5,9,13,17], axis=0)
    ax.plot(finger_connection_coordinates[:, 0], finger_connection_coordinates[:, 1], color='blue')
    # Set the title
    ax.set_title('Normalized Coordinates Plot')
    # Show the plot
    plt.show()

# Print an item in the HDF5 file
def print_hdf5_item(name, item):
    # Print the name and type of the item
    print(f'{name}: {type(item)}')
    # If the item is a dataset, print its shape and data
    if isinstance(item, h5py.Dataset):
        print(f'shape: {item.shape}')

# Store the hand landmarks in a HDF5 file
def store_hand_landmarks(dir_path, h5_path):

    # Create an HDF5 file to store the hand landmarks
    h5_file = h5py.File(h5_path, 'w')
    group = h5_file.create_group('hand_landmarks_group')
    print('HDF5 file created: ', h5_file.filename)
    print('HDF5 group created: ', group.name)

    # Iterate over the directories in the directory
    for dirname in os.listdir(dir_path):

        # Skip hidden directories and the HDF5 file
        if dirname.startswith('.') or dirname.endswith('.h5'):
            continue

        # Extract the label from the directory name
        label = int(dirname.split('_')[1])
        # Create the full directory path
        full_path = os.path.join(dir_path, dirname)
        # Create an empty array to store the hand landmarks
        data = np.empty((0, 21, 3))

        # Iterate over the files in the directory
        for filename in os.listdir(full_path):

            # Skip hidden files
            if filename.startswith('.'):
                continue

            # Create the full file path
            file_path = os.path.join(full_path, filename)
            # Extract the hand landmarks from the image at the file path
            features = process_image_with_hands(file_path)
            
            # Skip to the next iteration if no hand landmarks are detected
            if features is None:
                continue

            # Add the extracted features to the data array
            data = np.append(data, [features], axis=0)

        # Convert the data array to a NumPy array
        data = np.array(data)

        # Create a dataset in the HDF5 file
        dataset = group.create_dataset(dirname, data=data)
        # Set the label attribute of the dataset
        dataset.attrs['label'] = label
        # Print the dataset name, label, and shape
        print('Dataset created: ', dataset.name)
        print('Dataset label:', dataset.attrs['label'])
        print('Dataset shape:', dataset.shape)

    # Visit each item in the HDF5 file and print its name, type and shape
    print('Created HDF5 file:', h5_file.filename)
    h5_file.visititems(print_hdf5_item)

    # Close the HDF5 file
    h5_file.close()

# Define the directory paths
training_dir_path = '../data/train'
training_h5_path = '../data/train/training_data_demo.h5'
test_dir_path = '../data/test'
test_h5_path = '../data/test/test_data.h5'

# Store the training data
#store_hand_landmarks(training_dir_path, training_h5_path)
# Store the test data
#store_hand_landmarks(test_dir_path, test_h5_path)