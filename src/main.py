"""
Uses a trained model to classify the gestures performed by the detected hands in a video stream.
The video stream from the default webcam will be displayed, and the gestures performed by the detected hands
will be classified and displayed on the video.
The script assumes that the trained model is saved in a file named gesture_recognition_model.h5.
"""

# Import the necessary modules
import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp


# Import the drawing utilities and styles from the MediaPipe Hands module
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Print the docstring
print(__doc__)

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)

# Load the trained model
model = tf.keras.models.load_model("../models/gesture_recognition_model.h5")

# Predict the gesture performed by the detected hands in the video stream
def predict_gesture(hand_landmarks):

    # Convert the landmarks to a NumPy array
    coordinates = np.array([[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark])
    # Get the wrist landmark as the origin for the relative coordinates
    wrist = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z])
    # Calculate the relative coordinates and flip the hand
    relative_coordinates = -(coordinates - wrist)
    # Normalize the relative coordinates
    normalized_coordinates = (relative_coordinates - np.min(relative_coordinates)) / np.ptp(relative_coordinates)
    # Flatten the normalized coordinates
    flattened_coordinates = normalized_coordinates.flatten()

    # Predict the gesture performed by the hand
    prediction = model.predict(flattened_coordinates.reshape(1, -1))
    # Get the index of the predicted gesture
    predicted_gesture = np.argmax(prediction)
    # Get the confidence of the prediction
    confidence = prediction[0][predicted_gesture]
    # Print the gesture and confidence
    print("Predicted gesture: {}, confidence: {}".format(predicted_gesture, confidence))

# Use the Hands class from the MediaPipe Hands module to process the video frames
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    # Loop through the frames of the video
    while cap.isOpened():
    
        # Read a frame from the video
        success, image = cap.read()
        
        # If the frame is empty, skip it and print a message
        if not success:
          print("Ignoring empty camera frame.")
          continue
        
        # Make the image not writeable to improve performance
        image.flags.writeable = False
        # Convert the image to RGB color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Use the Hands object to process the image and detect hand landmarks
        results = hands.process(image)
        
        # Make the image writeable again and convert it to BGR color space
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # If hand landmarks were detected, draw them on the image
        if results.multi_hand_landmarks:

            # Iterate over the detected hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the detected hand landmarks on the image
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Predict the gesture performed by the hand
                predict_gesture(hand_landmarks)
                
        # Flip the image horizontally for a selfie-view display
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        
        # Check for user input to exit the program
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the capture and destroy the display window
cap.release()
cv2.destroyAllWindows()