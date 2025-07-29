from layer.Layer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, input):
        self.input = input
        # Explanation:
        # - self.input: the input of the layer
        # - self.activation: the activation function
        # - return: the output of the layer
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        # Explanation:
        # - output_gradient: the gradient of the output of the layer
        # - self.activation_derivative: the derivative of the activation function
        # - self.input: the input of the layer
        # - np.multiply: the element-wise multiplication of the output_gradient and the activation_derivative
        # - return: the gradient of the input of the layer
        return np.multiply(output_gradient, self.activation_derivative(self.input))
    
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x: np.ndarray) -> np.ndarray:
            """
            Calculate the sigmoid of a given input.
            """
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
            """
            Calculate the derivative of the sigmoid of a given input.
            """
            return sigmoid(x) * (1 - sigmoid(x))

        # Call the parent class constructor to initialize the activation and activation_derivative
        super().__init__(sigmoid, sigmoid_derivative)