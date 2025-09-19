import numpy as np
import matplotlib.pyplot as plt

from data.DataLoader import DataLoader
from layer.Convolutional import Convolutional
from layer.Activation import Sigmoid
from layer.Reshape import Reshape
from layer.Dense import Dense
from utils.Loss import Loss

class CnnDemo:

    @staticmethod
    def run():
        """
        Run the demo.
        """
        # Load and preprocess the data.
        X_train, Y_train, X_test, Y_test = DataLoader.load_and_preprocess_data()

        # Create the network.
        network = [
            Convolutional((1, 28, 28), 3, 5),       # Convolutional layer with 5 kernels of size 3x3
            Sigmoid(),                              # We always use an activation function after a convolutional layer
            Reshape((5, 26, 26), (5 * 26 * 26, 1)), # Reshape the output of the convolutional layer to a 2D array
            Dense(5 * 26 * 26, 2),                  # Dense layer with 2 output neurons (0 or 1)
            Sigmoid()                               # We always use an activation function after a dense layer
        ]

        # Train the network.
        CnnDemo.train(
            network,
            Loss.cross_entropy_loss,
            Loss.cross_entropy_loss_derivative,
            X_train,
            Y_train,
            epochs=20,
            learning_rate=0.015
        )

        # Test the accuracy of the network.
        CnnDemo.test_accuracy(network, X_test, Y_test)

    @staticmethod
    def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
        error_history = []
        for e in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                # forward
                output = CnnDemo.predict(network, x)

                # error
                error += loss(y, output)

                # backward
                grad = loss_prime(y, output)
                for layer in reversed(network):
                    grad = layer.backward(grad, learning_rate)

            error /= len(x_train)
            error_history.append(error)
            if verbose:
                print(f"{e + 1}/{epochs}, error={error}")
        return error_history

    @staticmethod
    def predict(network, input):
        output = input
        for layer in network:
            output = layer.forward(output)
        return output

    @staticmethod
    def test_accuracy(network, X_test, Y_test):
        """
        Test the accuracy of the network.
        """
        predictions = []
        actuals = []
        for x, y in zip(X_test, Y_test):
            output = CnnDemo.predict(network, x)
            # Convert output vector to a scalar value of 0 or 1
            predictions.append(np.argmax(output))
            actuals.append(np.argmax(y))

        # Calculate the confusion matrix
        true_positives = sum(1 for p, y in zip(predictions, actuals) if p == 1 and y == 1)
        true_negatives = sum(1 for p, y in zip(predictions, actuals) if p == 0 and y == 0)
        false_positives = sum(1 for p, y in zip(predictions, actuals) if p == 1 and y == 0)
        false_negatives = sum(1 for p, y in zip(predictions, actuals) if p == 0 and y == 1)
        confusion_matrix = np.array([[true_positives, false_positives], [false_negatives, true_negatives]])

        # Print the accuracy
        print(f"Accuracy: {(true_positives + true_negatives)*100 / len(X_test)}%")
        plt.matshow(confusion_matrix, cmap="Greens")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.xticks([0, 1], [0, 1], position=(0, -0.1))
        for (x, y), value in np.ndenumerate(confusion_matrix):
            plt.text(x, y, f"{value:.2f}", va="center", ha="center")
        plt.show()

if __name__ == "__main__":
    CnnDemo.run()