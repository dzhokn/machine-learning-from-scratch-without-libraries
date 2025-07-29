import numpy as np

class Loss:

    def cross_entropy_loss(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """
        Calculate the cross-entropy loss.

        Args:
            Y_true: np.ndarray, the true labels (e.g. [0, 1, 0, 1, 0])
            Y_pred: np.ndarray, the predicted labels (e.g. [0.1, 0.9, 0.1, 0.9, 0.1])

        Returns:
            float, the cross-entropy loss
        """
        e = 1e-7 # Avoid log(0)
        return -np.mean(Y_true * np.log(Y_pred + e) + (1 - Y_true) * np.log(1 - Y_pred + e))

    def cross_entropy_loss_derivative(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of the cross-entropy loss.

        Args:
            Y_true: np.ndarray, the true labels (e.g. [0, 1, 0, 1, 0])
            Y_pred: np.ndarray, the predicted labels (e.g. [0.1, 0.9, 0.1, 0.9, 0.1])

        Returns:
            np.ndarray, the derivative of the cross-entropy loss
        """
        e = 1e-7 # Avoid division by zero
        return ((1 - Y_true) / (1 - Y_pred + e) - Y_true / (Y_pred + e)) / np.size(Y_true) 