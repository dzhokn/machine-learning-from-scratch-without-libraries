from layer.Layer import Layer
from utils.Signal import Signal
import numpy as np

class Convolutional(Layer):

    def __init__(self, input_shape: tuple, kernel_size: int, num_of_kernels: int):
        """
        Initialize the convolutional layer.
        The layer will be initialized with random weights and biases.

        The shape of the output is calculated as:
         * output_shape = (num_of_kernels, input_height - kernel_size + 1, input_width - kernel_size + 1)
         * kernel_shape = (num_of_kernels, input_depth, kernel_size, kernel_size)
         * biases_shape = (output_shape)

        Args:
            input_shape: tuple, shape of the input data (depth, height, width)
            kernel_size: int, size of the kernel (e.g. 4 for 4x4 kernel)
            num_of_kernels: int, number of kernels in the layer (e.g. 2 for 2 kernels)
        """
        input_depth, input_height, input_width = input_shape                            # (e.g. 3, 28, 28)
        self.num_of_kernels = num_of_kernels                                            # (e.g. 2)
        self.input_shape = input_shape                                                  # (e.g. 3, 28, 28)
        self.kernel_shape = (num_of_kernels, input_depth, kernel_size, kernel_size)     # (e.g. 2, 3, 4, 4)
        self.output_shape = (num_of_kernels, input_height - kernel_size + 1, input_width - kernel_size + 1) # (e.g. 2, 25, 25)

        # Generate random weights and biases
        self.kernels = np.random.randn(*self.kernel_shape)  # (e.g. 2, 3, 4, 4)
        self.biases = np.random.randn(*self.output_shape)   # (e.g. 2, 25, 25)

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Forward pass through the convolutional layer.
            - The input is a 3D tensor (e.g. image tensor).
            - The output is a 3D tensor (e.g. feature map).

        The formula is:
            - Y_i = B_i + \\sum_{j=1}^{n}{X_j \\star K_{ij}}
        where:
            - Y_i is the output of the i-th kernel
            - B_i is the bias of the i-th kernel
            - X_j is the j-th depth of the input
            - K_{ij} is the j-th depth of the i-th kernel
            - n is the number of depths of the input

        Args:
            input: np.ndarray, input tensor (e.g. image tensor)
        Returns:
            np.ndarray, output tensor (e.g. feature map)
        """
        self.input = input
        self.output = np.copy(self.biases) # Output = bias + sum(input * kernel)

        # For each kernel (e.g. 2)
        for i in range(self.num_of_kernels): 
            # For each depth of the input (e.g. 3)
            for j in range(self.input_shape[0]):
                # The formula is:
                # - output[i] += input[j] * kernels[i, j]
                # where:
                # - output[i] is the output of the i-th kernel
                # - input[j] is the j-th depth of the input
                # - kernels[i, j] is the j-th depth of the i-th kernel
                # - mode='valid' is the mode of the correlation operation
                self.output[i] += Signal.correlate2d(self.input[j], self.kernels[i, j], mode='valid')

        # Return the output of the layer
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass through the convolutional layer.

        Args:
            output_gradient: np.ndarray, the gradient of the output (e.g. [[1, 2, 3], [4, 5, 6]])
            learning_rate: float, the learning rate

        Returns:
            np.ndarray, the gradient of the input (e.g. [1, 2, 3, 4, 5, 6])
        """
        kernels_gradient = np.zeros(self.kernel_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.num_of_kernels):
            for j in range(self.input_shape[0]): # input depth
                kernels_gradient[i, j] = Signal.correlate2d(self.input[j], output_gradient[i], mode='valid')
                input_gradient[j] += Signal.convolve2d(output_gradient[i], self.kernels[i, j], mode='full')

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient

        return input_gradient