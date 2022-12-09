"""
This script preprocesses the hand landmarks detected in a static image using MediaPipe Hands. 
The landmarks are converted to relative coordinates which are normalized and flattened. 
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
def preprocess_data(file_path):

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        
            # Read an image, flip it around y-axis for correct handedness output
            image = cv2.flip(cv2.imread(file_path), 1)

            # Convert the BGR image to RGB before processing
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print the handedness of the detected hands and draw hand landmarks on the image
            print('Handedness:', results.multi_handedness)

            # If no hand landmarks are detected, skip to the next iteration
            if not results.multi_hand_landmarks:
                return None, None

            # Iterate over the detected hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:

                # Convert the landmarks to a NumPy array
                coordinates = np.array([[lmk.x, lmk.y] for lmk in hand_landmarks.landmark])

                # Get the wrist landmark as the origin for the relative coordinates
                wrist = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y])

                # Calculate the relative coordinates and flip the hand
                relative_coordinates = -(coordinates - wrist)

                # Normalize the relative coordinates
                normalized_coordinates = (relative_coordinates - np.min(relative_coordinates)) / np.ptp(relative_coordinates)

                # Flatten the normalized coordinates
                flattened_coordinates = normalized_coordinates.flatten()

                return flattened_coordinates, normalized_coordinates

# Plot the normalized coordinates
def plot(normalized_coordinates):

    # Plot the normalized coordinates
    fig = plt.figure()
    ax = fig.add_subplot()
    # Plot the points
    ax.scatter(normalized_coordinates[:, 0], normalized_coordinates[:, 1])
        # Thumb plot
    thumb_coordinates = normalized_coordinates.take([0,1,2,3,4], axis=0)
    ax.plot(thumb_coordinates[:, 0], thumb_coordinates[:, 1], color='blue')
    # Index finger plot
    index_finger_coordinates = normalized_coordinates.take([0,5,6,7,8], axis=0)
    ax.plot(index_finger_coordinates[:, 0], index_finger_coordinates[:, 1], color='blue')
    # Middle finger plot
    middle_finger_coordinates = normalized_coordinates.take([9,10,11,12], axis=0)
    ax.plot(middle_finger_coordinates[:, 0], middle_finger_coordinates[:, 1], color='blue')
    # Ring finger plot
    ring_finger_coordinates = normalized_coordinates.take([13,14,15,16], axis=0)
    ax.plot(ring_finger_coordinates[:, 0], ring_finger_coordinates[:, 1], color='blue')
    # Pinky plot
    pinky_coordinates = normalized_coordinates.take([0,17,18,19,20], axis=0)
    ax.plot(pinky_coordinates[:, 0], pinky_coordinates[:, 1], color='blue')
    # Finger connection plot
    finger_connection_coordinates = normalized_coordinates.take([5,9,13,17], axis=0)
    ax.plot(finger_connection_coordinates[:, 0], finger_connection_coordinates[:, 1], color='blue')
    # Set the title
    ax.set_title('Normalized Coordinates Plot')
    # Show the plot
    plt.show()

# Create an HDF5 file to store the hand landmarks
h5_file = h5py.File('../data/train/hand_landmarks.h5', 'w')
group = h5_file.create_group('hand_landmarks_group')
train_dir = '../data/train'
data = np.empty((0, 42))

for dirname in os.listdir(train_dir):

    if not dirname.startswith('label'):
        continue

    label = int(dirname.split('_')[1])

    dir_path = os.path.join(train_dir, dirname)

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        print(file_path)
        features, norms = preprocess_data(file_path)
        if features is None:
            continue
        plot(norms)    
        print('Features shape:', features.shape)
        data = np.append(data, [features], axis=0)

    data = np.array(data)
    print('Data shape:', data.shape)

    # Create a dataset in the HDF5 file
    dataset = group.create_dataset(dirname, data=data)
    print('Dataset shape:', dataset.shape)
    print('label:', label)
    dataset.attrs['label'] = label

# Close the HDF5 file
h5_file.close()

