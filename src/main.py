"""
The main.py script uses a trained model to classify the gestures performed by the detected hands in a video stream.
The video stream from the default webcam will be displayed, and the gestures performed by the detected hands
will be classified and displayed on the video.
The script assumes that the trained model is available in  models/model_main.h5.
"""

# Import the necessary modules
import os
import cv2
import model
import mediapipe as mp

from model import predict_gesture

# Import the drawing utilities and styles from the MediaPipe Hands module
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Print the docstring
print(__doc__)

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)

# Load the trained model
model_path = '../models/model_main.h5'
gesture_recognition_model = model.load_model(model_path)

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
        
        def handmarks():
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
                    predict_gest, confidence = predict_gesture(gesture_recognition_model, hand_landmarks)
                    print(predict_gest, confidence)
                    return predict_gest, confidence

        def draw_landmarks():
             for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

        # Flip the image horizontally for a selfie-view display
        try:
            pred_gest, confidence = handmarks()
        except:
            pred_gest = None
            confidence = None
            pass
        ret, frame = cap.read()

        try:
            # Search the path of the Logo 
            logo = cv2.imread(('../emojis/{}.png').format(pred_gest))
            
            # Resize the logo and position it
            logo = cv2.resize(logo, (100, 100))
            frame_height, frame_width = frame.shape[:2]
            start_x = int(frame_width * 0.50)
            start_y = int(frame_height * 0.05)

            # Insert Emoji into camera feed
            frame[start_y:start_y+100, start_x:start_x+100] = logo
            
            # Draw the landmarks
            draw_landmarks()
            frame = cv2.flip(frame, 1)
            print("we after frame")
            
            cv2.putText(frame, "predicted gesture "+str(pred_gest) +" "+ str(round(confidence*100, 1))+"%", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                        
        except Exception as G:
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "No gesture detected", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        cv2.imshow('MediaPipe Hands', frame)
        
        # Check for user input to exit the program
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the capture and destroy the display window
cap.release()
cv2.destroyAllWindows()

def get_emojis():
    emojis_folder = './emojis'
    emojis = []
    for emoji in range(len(os.listdir(emojis_folder))):
        print(emoji)
        emojis.append(cv2.imread(emojis_folder+str(emoji)+'.png', -1))
    return emojis
