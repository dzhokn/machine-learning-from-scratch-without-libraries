from layer.Layer import Layer
import numpy as np

class MaxPool(Layer):
    def __init__(self, pool_size: int, stride: int):
        self.pool_size = pool_size
        self.stride = stride
        self.max_val_indices = None
        self.input_shape = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        # Get the dimensions of the input
        num_filters, h, w = input.shape

        # Define the output dimensions
        h_out = (h - self.pool_size) // self.stride + 1
        w_out = (w - self.pool_size) // self.stride + 1

        # Store the indices of the max values here. They are needed for the backward propagation.
        self.max_val_indices = np.zeros((num_filters, h_out, w_out, 2))
        self.input_shape = input.shape

        # Initialize the output
        output = np.zeros((num_filters, h_out, w_out))
        # Iterate over the input
        for h_out_idx in range(h_out):
            for w_out_idx in range(w_out):
                # Get the pool
                h_start = h_out_idx*self.stride
                h_end = h_start + self.pool_size
                w_start = w_out_idx*self.stride
                w_end = w_start + self.pool_size
                pool = input[:, h_start:h_end, w_start:w_end]
                # Get the max values for that pool (the number of max values == the number of filters)
                max_values = np.max(pool, axis=(1, 2))
                output[:, h_out_idx, w_out_idx] = max_values
                # NB: Remember the indices for these max values. These indices will be used for the backward propagation.
                self.set_max_val_indices(pool, h_out_idx, w_out_idx, max_values)
        # Return the output
        return output

    def set_max_val_indices(self, pool: np.ndarray, h_out_idx: int, w_out_idx: int, max_values: np.ndarray) -> np.ndarray:
        """ We can't use np.argmax since it axis parameter accepts only a single value, and not a tuple of values. """
        # For each filter
        for filter_idx in range(pool.shape[0]):
            # For each row
            for pool_h_idx in range(pool.shape[1]):
                # For each column
                for pool_w_idx in range(pool.shape[2]):
                    # IMPORTANT:We store the indices of the max values within a pool.
                    if pool[filter_idx, pool_h_idx, pool_w_idx] == max_values[filter_idx]:
                        self.max_val_indices[filter_idx, h_out_idx, w_out_idx, 0] = pool_h_idx
                        self.max_val_indices[filter_idx, h_out_idx, w_out_idx, 1] = pool_w_idx

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Get the dimensions of the output
        num_filters, h_out, w_out = output_gradient.shape
        # Initialize the input gradient by setting all values to zero by default.
        input_gradient = np.zeros(self.input_shape)

        # Iterate over the output. For each value in the output matrix, we update a single value in the input matrix.
        for filter_idx in range(num_filters):
            for h_out_idx in range(h_out):
                for w_out_idx in range(w_out):
                    # Get the index of the maximum value within a pool.
                    max_val_pool_indices = self.max_val_indices[filter_idx, h_out_idx, w_out_idx]
                    pool_h_idx, pool_w_idx = max_val_pool_indices[0], max_val_pool_indices[1]
                    # Calculate the index of the maximum value in the input matrix.
                    input_h_idx = int(h_out_idx*self.stride + pool_h_idx)
                    input_w_idx = int(w_out_idx*self.stride + pool_w_idx)
                    # Update only the value at that index. All other values remain zero.
                    input_gradient[filter_idx, input_h_idx, input_w_idx] = output_gradient[filter_idx, h_out_idx, w_out_idx]
        # Return the input gradient
        return input_gradient