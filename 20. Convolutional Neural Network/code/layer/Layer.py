import numpy as np

class Layer:
    """
    Base (abstract) class for all layers.
    """
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        # Return layer's output
        pass

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Update parameters and return the input gradient
        pass