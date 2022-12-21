# GestureDetection_ML_ITA

This project uses a neural network to recognize gestures from hands in a live video feed. The recognized gestures are used to play a simple game.

## Usage

In general, it is recommended to clone this repository to your Google Drive for an easy integration with Google Colab.

To train the model, you need to download [the train and test datasets](https://dhbwstg-my.sharepoint.com/:f:/g/personal/inf20111_lehre_dhbw-stuttgart_de/EkzpxgUaZn9FhQzTylc5D-8B0XFuU4BwawXWmELFV0OezA?e=uR1XDG) and place them in the `data/train` and `data/test` directories.

To train the model, there are two methods:

1. Run the following commands:

```
cd src
python train_model.py
```

2. Use the `train_model.ipynb` notebook in your Google Colab GPU runtime.

The trained model is stored in `models/model_<hash>.h5`. The training results are stored in `models/results/results_<hash>.md`. 

Note that a perfectly fitted pretrained model already exists located in `models/model_main.h5`. Therefore, starting the training process is recommended only for your own research purposes. 

To start the application, run the following commands:

```
cd src
python main.py
```

The video stream from the default webcam will be displayed, and the gestures performed by the detected hands will be used to control the game.

## Requirements

The project requires the following libraries:

- `mediapipe`
- `tensorflow`
- `opencv-python`
- `matplotlib`
- `numpy`
- `h5py`
- `scikeras`
- `scikit-optimize`
- `pandas`
- `tabulate`

To install these libraries globally, run the following command (**unrecommended**):

```
pip install -r requirements.txt
```

To install these libraries in a conda environment, run the following command (**recommended**):

```
conda create --name <thisproject>
conda activate <thisproject>
pip install -r requirements.txt
```

## Troubleshooting

If you encounter the `ImportError: cannot import name 'builder' from 'google.protobuf.internal error'`, it is likely because your `tensorflow` requires an older version of `protobuf` that does not include the `builder.py` script, but your `mediapipe` requires a newer version of `protobuf` that includes this script.

To resolve this issue, you can try the following method:

Adapted from [this Stack Overflow answer](https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal)

1. Install the latest version of the `protobuf` that includes the `builder.py` script. To do this, run the following command:

```
pip install protobuf --upgrade
```

2. Copy `builder.py` from `...\site-packages\google\protobuf\internal` to your computer (let's say 'Documents')

3. Uninstall the latest version of `protobuf`. To do this, run the following command:

```
pip uninstall protobuf
```

4. Install a version of `protobuf` that is compatible with `tensorflow` (e.g. 3.19.6). To do this, run the following command:

```
pip install protobuf==<version-number>
```

5. Copy `builder.py` from (let's say 'Documents') to `...\site-packages\google\protobuf\internal`

6. Run your code
