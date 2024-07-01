
# MNIST Digit Classification

## Description

This project uses a neural network to classify handwritten digits from the MNIST dataset. The MNIST dataset is a well-known dataset in the machine learning community, consisting of 70,000 28x28 grayscale images of handwritten digits (0-9). The goal is to build and train a neural network to accurately classify these digits.

## Features

- **Data Loading and Preprocessing**:
  - Load the MNIST dataset.
  - Normalize the pixel values to be between 0 and 1.

- **Model Building**:
  - Build a neural network with one hidden layer using TensorFlow and Keras.

- **Model Training**:
  - Train the model on the training data with validation.

- **Model Evaluation**:
  - Evaluate the model's performance using the test data.
  - Plot training and validation accuracy and loss.

## Prerequisites

- Python 3.x
- TensorFlow
- Matplotlib

## Installation

1. Clone the repository or download the `mnist_digit_classification.py` file.
    ```sh
    git clone https://github.com/yourusername/mnist-digit-classification.git
    cd mnist-digit-classification
    ```

2. Install the required libraries using pip:
    ```sh
    pip install tensorflow matplotlib
    ```

## Usage

1. Run the application by executing the following command in the terminal or command prompt:
    ```sh
    python mnist_digit_classification.py
    ```

2. The script will load the MNIST dataset, build the neural network, train the model, and evaluate its performance. Training and validation accuracy and loss will be plotted.

## Code Overview

The main components of the script include:

- **Data Loading and Preprocessing**:
  - The MNIST dataset is loaded using `datasets.mnist.load_data()`.
  - The pixel values of the images are normalized to the range [0, 1].

- **Building the Model**:
  - The model consists of three layers:
    - `Flatten`: Converts each 28x28 image into a 784-element vector.
    - `Dense`: A fully connected layer with 128 neurons and ReLU activation.
    - `Dense`: The output layer with 10 neurons (one for each digit) and softmax activation.

- **Compiling the Model**:
  - The model is compiled with the Adam optimizer, sparse categorical crossentropy loss, and accuracy as a metric.

- **Training the Model**:
  - The model is trained for 10 epochs with 20% of the training data used for validation.

- **Evaluating the Model**:
  - The model's performance is evaluated on the test set, and the test accuracy is printed.
  - The training and validation accuracy and loss are plotted.

## Example

Here's a sample code snippet for `mnist_digit_classification.py`:

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Step 2: Build the neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),       # Flatten the 28x28 images into a 784-element vector
    layers.Dense(128, activation='relu'),       # Fully connected layer with 128 neurons and ReLU activation
    layers.Dense(10, activation='softmax')      # Output layer with 10 neurons (one for each digit) and softmax activation
])

# Step 3: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_split=0.2)

# Step 5: Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test
