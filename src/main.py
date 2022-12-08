"""
Uses a trained model to classify the gestures performed by the detected hands in a video stream.

To use the model, run the script as follows:
    python main.py

The video stream from the default webcam will be displayed, and the gestures performed by the detected hands
will be classified and displayed on the video.

The script assumes that the trained model is saved in a file named gesture_recognition_model.h5.
"""

# Import the necessary modules
import cv2
import mediapipe as mp

# Import the drawing utilities and styles from the MediaPipe Hands module
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Print the docstring
print(__doc__)

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)

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

        # Flip the image horizontally for a selfie-view display
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        
        # Check for user input to exit the program
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the capture and destroy the display window
cap.release()
cv2.destroyAllWindows()