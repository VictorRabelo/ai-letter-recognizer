# AI Letter Recognizer

This project is a simple OCR application that lets you draw English letters (A-Z, a-z) with your mouse. A neural network trained on the EMNIST dataset recognizes the drawn letter and displays the prediction with a confidence score.

## Features

- Tkinter-based drawing interface.
- Conversion of drawings to 28x28 grayscale images.
- Convolutional neural network trained on EMNIST letters.
- Model is saved to disk and loaded automatically.
- Clear button to reset the canvas.

## Installation

1. **Clone the repository** and install the required Python packages:

```bash
pip install tensorflow tensorflow-datasets pillow
```

TensorFlow may require additional system dependencies depending on your platform. Refer to the [TensorFlow installation guide](https://www.tensorflow.org/install) if you encounter issues.

## Dataset

The model uses the `emnist/byclass` dataset provided by [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/emnist). The dataset is downloaded automatically when training the model for the first time.

## Training the Model

If `emnist_letters.h5` is not present in the project directory, running the application will trigger training automatically. Training can also be started manually:

```bash
python model.py
```

Training may take several minutes depending on your hardware and will download the EMNIST dataset if not already cached.

## Running the Application

Launch the interface with:

```bash
python main.py
```

Draw a letter on the canvas, release the mouse button, and the predicted character with its confidence will appear below the canvas. Use the **Clear** button to draw a new letter.

## Project Structure

- `interface.py` – GUI logic for drawing and capturing images.
- `model.py` – Training, loading, and prediction utilities for the neural network.
- `main.py` – Entry point that ties the interface and model together.

## License

This project is provided for educational purposes and can be freely used and modified.
