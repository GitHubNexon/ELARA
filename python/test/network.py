import numpy as np
import os
import argparse
from PIL import Image
import tqdm

# Define dataset path
DATASET_PATH = "C:/ELARA/python/data/MNIST/"

# Load MNIST dataset
def load_mnist():
    def load_file(filename, offset, shape=None):
        with open(os.path.join(DATASET_PATH, filename), "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=offset)
        return data if shape is None else data.reshape(shape) / 255.0
    
    X_train = load_file("train-images.idx3-ubyte", 16, (-1, 28 * 28))
    y_train = load_file("train-labels.idx1-ubyte", 8)
    X_test = load_file("t10k-images.idx3-ubyte", 16, (-1, 28 * 28))
    y_test = load_file("t10k-labels.idx1-ubyte", 8)
    
    return X_train, y_train, X_test, y_test

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# One-hot encoding
def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

# Initialize weights and biases
def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

# Backpropagation
def backpropagation(X, y, Z1, A1, Z2, A2, W1, b1, W2, b2, learning_rate):
    m = X.shape[0]
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    return W1, b1, W2, b2

# Training function
def train(X_train, y_train, hidden_size=64, epochs=10, learning_rate=0.1):
    input_size, output_size = X_train.shape[1], 10
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)
    y_train_encoded = one_hot_encode(y_train)
    
    for epoch in range(epochs):
        for i in tqdm.tqdm(range(0, len(X_train), 64), desc=f"Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_train[i:i+64], y_train_encoded[i:i+64]
            Z1, A1, Z2, A2 = forward_propagation(X_batch, W1, b1, W2, b2)
            W1, b1, W2, b2 = backpropagation(X_batch, y_batch, Z1, A1, Z2, A2, W1, b1, W2, b2, learning_rate)
    
    return W1, b1, W2, b2

# Prediction function
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1)

# Function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path)
    
    img = img.convert("L")
    
    img = img.resize((28, 28))
    
    img = np.array(img) / 255.0
    
    img = img.flatten().reshape(1, -1)
    
    return img

def main(args):
    X_train, y_train, X_test, y_test = load_mnist()

    W1, b1, W2, b2 = train(X_train, y_train)

    predictions = predict(X_test, W1, b1, W2, b2)

    accuracy = np.mean(predictions == y_test)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    if args.image:
        img = preprocess_image(args.image)
        prediction = predict(img, W1, b1, W2, b2)
        print(f"Predicted number for the uploaded image: {prediction[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Network for MNIST")
    parser.add_argument('--image', type=str, help="Path to the image file for prediction")
    args = parser.parse_args()

    main(args)

#python network.py --image "path_to_image_file.png"
