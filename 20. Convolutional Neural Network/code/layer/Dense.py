from layer.Layer import Layer
import numpy as np

class Dense(Layer):
    """
    Dense (fully-connected) layer.
    """
    def __init__(self, input_size, output_size):
        # Initialize random weights and biases
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        # Return the standard output of the layer (weights * input + bias)
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Calculate the weights gradient.
        weights_gradient = np.dot(output_gradient, self.input.T)
        # Calculate the input gradient
        input_gradient = np.dot(self.weights.T, output_gradient)
        # Update the weights and bias
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient