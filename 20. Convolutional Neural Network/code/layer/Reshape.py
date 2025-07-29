from layer.Layer import Layer
import numpy as np

class Reshape(Layer):
    """
    Reshape layer.
    It reshapes the input to the output shape.
    It is used to reshape the output of a convolutional layer to a 2D array.
    It is also used to reshape the output of a dense layer to a 1D array.
    """

    def __init__(self, input_shape: tuple, output_shape: tuple):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Reshape the input to the output shape.

        Args:
            input: np.ndarray, the input to be reshaped (e.g. [1, 2, 3, 4, 5, 6])

        Returns:
            np.ndarray, the reshaped input (e.g. [[1, 2, 3], [4, 5, 6]])
        """
        return input.reshape(self.output_shape)

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Reshape the output gradient to the input shape.

        Args:
            output_gradient: np.ndarray, the gradient of the output (e.g. [[1, 2, 3], [4, 5, 6]])

        Returns:
            np.ndarray, the gradient of the input (e.g. [1, 2, 3, 4, 5, 6])
        """
        return output_gradient.reshape(self.input_shape)