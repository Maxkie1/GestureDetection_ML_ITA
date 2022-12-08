# GestureDetection_ML_ITA

This project uses a convolutional neural network to recognize gestures from images of hands in a live video feed. The recognized gestures are used to play a simple game.

## Usage

To train the model, run the following command:

```
python gesture_recognition_model.py
```

The trained model will be saved to a file named `gesture_recognition_model.h5`.

To start the application, run the following command:

```
python main.py
```

The video stream from the default webcam will be displayed, and the gestures performed by the detected hands will be used to control the game.

## Requirements

The project requires the following libraries:

- `mediapipe`
- `tensorflow`
- `opencv-python`

To install these libraries, run the following commands:

```
pip install mediapipe tensorflow opencv-python
```

These libraries need to be installed before running the `gesture_recognition_model.py` and `main.py` scripts.

## Troubleshooting

If you encounter the `ImportError: cannot import name 'builder' from 'google.protobuf.internal error'`, it is likely because your `tensorflow` requires an older version of the `protobuf` that does not include the `builder.py` script, but your `mediapipe` requires a newer version of the `protobuf` library that includes this script.

To resolve this issue, you can try the following method:

1. Install the latest version of the protobuf library that includes the builder.py script. To do this, run the following command:

```
pip install protobuf --upgrade
```

2. Copy builder.py from `...\site-packages\google\protobuf\internal` to your computer (let's say 'Documents')

3. Uninstall the latest version of `protobuf`. To do this, run the following command:

```
pip uninstall protobuf
```

4. Install the version of the `protobuf` that is compatible with your `tensorflow`. To do this, run the following command:

```
pip install protobuf==[version-number]
```

Replace `[version-number]` with the version number of the `protobuf` that is compatible with your `tensorflow`.

5. Copy `builder.py` from (let's say 'Documents') to `...\site-packages\google\protobuf\internal`

6. Run your code