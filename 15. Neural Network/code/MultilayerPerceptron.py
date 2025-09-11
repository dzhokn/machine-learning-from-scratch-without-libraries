import numpy as np
import matplotlib.pyplot as plt

class MultilayerPerceptron:
    '''
    Multilayer Perceptron class to train a multilayer perceptron model with three layers: input, hidden and output.
    '''
    # Attributes
    weights: list[np.ndarray]
    biases: list[np.ndarray]

    # Constructor
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        '''
        Initialize the Multilayer Perceptron object.

        Args:
            input_size          : int - number of input nodes
            hidden_size         : int - number of hidden nodes
            output_size         : int - number of output nodes
        '''

        # Initialize weights using random numbers
        self.weights = [np.random.randn(hidden_size, input_size), np.random.randn(output_size, hidden_size)]
        # Initialize biases to 0
        self.biases = [np.zeros((hidden_size, 1)), np.zeros((output_size, 1))]

    # Activation function
    def sigmoid(self, Z: np.ndarray) -> np.ndarray:
        '''
        Calculate the sigmoid of the weighted input.

        Args:
            Z                   : np.ndarray - input with shape (layer_size, m), where m is the number of observations

        Returns:
            sigmoid             : np.ndarray - sigmoid of the input with shape (layer_size, m)
        '''
        return 1 / (1 + np.exp(-Z))

    # Activation function derivative
    def sigmoid_derivative(self, Z: np.ndarray) -> np.ndarray:
        '''
        Calculate the derivative of the sigmoid function.

        Args:
            Z                   : np.ndarray - input with shape (layer_size, m), where m is the number of observations

        Returns:
            sigmoid_derivative  : np.ndarray - derivative of the sigmoid function with shape (layer_size, m)
        '''
        return Z * (1 - Z)

    def forward(self, X: np.ndarray) -> list[np.ndarray]:
        '''
        Forward pass through the network.

        Args:
            X                   : np.ndarray - input with shape (m, n), where m is the number of observations 
                                               and n is the number of features (input_size)

        Returns:
            a1                  : np.ndarray - activation with shape (hidden_size, m)
            a2                  : np.ndarray - activation with shape (output_size, m)
        '''
        # z1 = w1 * X.T + b1
        # z1 = (hidden_size, input_size) * (input_size, m) + (hidden_size, 1)
        # z1 = (hidden_size, m)
        z1 = self.weights[0] @ X.T + self.biases[0]
        # The shape of a1 is (hidden_size, m).
        a1 = self.sigmoid(z1)

        # z2 = w2 * a1 + b2
        # z2 = (output_size, hidden_size) * (hidden_size, m) + (output_size, 1)
        z2 = self.weights[1] @ a1 + self.biases[1]
        # The shape of a2 is (output_size, m).
        a2 = self.sigmoid(z2)

        return [a1, a2]

    def backward(self, X: np.ndarray, Y: np.ndarray, activations: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Backward pass through the network. Here is where the magic happens.

        Args:
            X                   : np.ndarray - input with shape (m, n), where m is the number of observations 
                                               and n is the number of features (input_size)
            Y                   : np.ndarray - output with shape (m, 1)
            activations         : list[np.ndarray] - list of activations with shape (layer_size, m)

        Returns:
            dc_dw1              : np.ndarray - derivative of the error w.r.t. the weights of the hidden layer with shape (hidden_size, input_size)
            dc_dw2              : np.ndarray - derivative of the error w.r.t. the weights of the output layer with shape (output_size, hidden_size)
            dc_db1              : np.ndarray - derivative of the error w.r.t. the biases of the hidden layer with shape (hidden_size, 1)
            dc_db2              : np.ndarray - derivative of the error w.r.t. the biases of the output layer with shape (output_size, 1)
        '''
        a1 = activations[0] # The shape of a1 is (hidden_size, m).
        a2 = activations[1] # The shape of a2 is (output_size, m).

        # Calculate the error
        error = (a2 - Y.T) ** 2 # The shape of error is (output_size, m) and the shape of Y.T is (output_size, m)
        m = len(Y) # The number of observations

        # The shape of dc_da2 is (output_size, m)
        dc_da2 = 2 * (a2 - Y.T)                             # Derivative of the error w.r.t. the activation
        # The shape of dc_dz2 is (output_size, m)
        dc_dz2 = self.sigmoid_derivative(a2) * dc_da2       # Derivative of the error w.r.t. the delta of the output layer
        # The shape of dc_dw2 is (output_size, hidden_size)
        dc_dw2 = np.dot(dc_dz2, a1.T) / m                   # Derivative of the error w.r.t. the weights of the output layer
        # The shape of dc_db2 is (output_size, 1)
        dc_db2 = np.sum(dc_dz2, axis=1, keepdims=True) / m  # Derivative of the error w.r.t. the biases of the output layer

        # The shape of dc_da1 is (hidden_size, m)
        dc_da1 = self.weights[1].T @ dc_dz2                 # Derivative of the error w.r.t. the activation of the hidden layer
        # The shape of dc_dz1 is (hidden_size, m)
        dc_dz1 = self.sigmoid_derivative(a1) * dc_da1       # Derivative of the error w.r.t. the delta of the hidden layer
        # The shape of dc_dw1 is (hidden_size, input_size)
        dc_dw1 = np.dot(dc_dz1, X) / m                      # Derivative of the error w.r.t. the weights of the hidden layer
        # The shape of dc_db1 is (hidden_size, 1)
        dc_db1 = np.sum(dc_dz1, axis=1, keepdims=True) / m  # Derivative of the error w.r.t. the biases of the hidden layer

        return dc_dw1, dc_dw2, dc_db1, dc_db2, error

    def update_weights_and_biases(self, dc_dw1: np.ndarray, dc_dw2: np.ndarray, dc_db1: np.ndarray, dc_db2: np.ndarray, learn_rate: float):
        '''
        Update the weights and biases of the network. The learning rate is a hyperparameter that controls the step size of the gradient descent.

        Args:
            dc_dw1              : np.ndarray - derivative of the error w.r.t. the weights of the hidden layer with shape (hidden_size, input_size)
            dc_dw2              : np.ndarray - derivative of the error w.r.t. the weights of the output layer with shape (output_size, hidden_size)
            dc_db1              : np.ndarray - derivative of the error w.r.t. the biases of the hidden layer with shape (hidden_size, 1)
            dc_db2              : np.ndarray - derivative of the error w.r.t. the biases of the output layer with shape (output_size, 1)
            learn_rate          : float - learning rate
        '''
        self.weights[0] -= learn_rate * dc_dw1
        self.weights[1] -= learn_rate * dc_dw2
        self.biases[0] -= learn_rate * dc_db1
        self.biases[1] -= learn_rate * dc_db2
    
    def train(self, X: np.ndarray, Y: np.ndarray, learn_rate: float, epochs: int) -> list[float]:
        '''
        Train the network.

        Args:
            X                   : np.ndarray - input with shape (m, n), where m is the number of observations 
                                               and n is the number of features (input_size)
            Y                   : np.ndarray - output with shape (m, 1)
            learn_rate          : float - learning rate
            epochs              : int - number of epochs

        Returns:
            losses              : list[float] - list of losses for each epoch
        '''
        losses = []
        for epoch in range(epochs):
            activations = self.forward(X)

            dc_dw1, dc_dw2, dc_db1, dc_db2, error = self.backward(X, Y, activations)
            losses.append(np.sum(error))

            # Update the weights and biases
            self.update_weights_and_biases(dc_dw1, dc_dw2, dc_db1, dc_db2, learn_rate)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch} error: {round(np.sum(error), 3)}")
        return losses

# Demo of training the Multilayer Perceptron model with data about salaries, years of experience and level of education
def main():
    # X1 = years of experience
    X1 = [1.2, 1.3, 1.5, 1.8, 2, 2.1, 2.2, 2.5, 2.8, 2.9, 3.1, 3.3, 3.5, 3.8, 4, 4.1, 4.5, 4.9, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10, 11, 12, 13, 14, 15]
    # X2 = level of education
    X2 = [2, 5, 3, 5, 3, 4, 2, 3, 4, 4, 3, 7, 5, 6, 5, 5, 2, 3, 4, 5, 6, 7, 5, 3, 2, 4, 5, 7, 3, 5, 7, 7, 5]
    # Y = salary
    Y = [2900, 3300, 3100, 4200, 3500, 3800, 3300, 3500, 3750, 4000, 3900, 5300, 4420, 5000, 4900, 5200, 3900, 4800, 5700, 6500, 6930, 7500, 7360, 6970, 6800, 7500, 8000, 9500, 11000, 9500, 12300, 13700, 12500]

    # Pandas data frame works with vectorized arrays (33 arrays of 1 element each)
    vectorized_X1 = np.array(X1).reshape(-1, 1)
    vectorized_X2 = np.array(X2).reshape(-1, 1)
    Y_train = np.array(Y).reshape(-1, 1) / 20000  # We divide by 20000 to scale down the output (it must be between 0 and 1)

    # Build the data frame
    X_train = np.concatenate([vectorized_X1, vectorized_X2], axis=1)

    network = MultilayerPerceptron(2, 2, 1)  # 2 input nodes, 2 hidden nodes, 1 output node
    losses = network.train(X_train, Y_train, 0.5, 10000)

    # Plot training loss
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    main()