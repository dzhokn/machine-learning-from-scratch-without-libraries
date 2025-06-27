import numpy as np

class Perceptron:
    '''
    Perceptron class to train a perceptron model.
    '''

    # Attributes
    learning_rate   : float
    n_iterations    : int
    w               : np.ndarray
    b               : float

    # Constructor
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        '''
        Initialize the Perceptron object.

        Args:
            learning_rate       : float - learning rate
            n_iterations        : int - number of iterations (epochs)
        '''
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = np.zeros(2)
        self.b = 0

    def weighted_sum(self, x: np.ndarray) -> float:
        '''
        Calculate the weighted sum of the input features - x_1 * w_1 + x_2 * w_2 + ... + x_n * w_n + b.

        Args:
            x                   : np.ndarray - a single observation with shape (n, )

        Returns:
            weighted_sum        : float - weighted sum
        '''
        return np.dot(x, self.w) + self.b
    
    def predict(self, x: np.ndarray) -> float:
        '''
        Predict the binary output for single observation.

        Args:
            x                   : np.ndarray - a single observation with shape (n, )

        Returns:
            y_pred              : float - predicted output (0 or 1)
        '''
        H = self.weighted_sum(x)
        return np.where(H >= 0.5, 1, 0) # Binary classification (0 or 1)
    
    def calculate_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        '''
        Calculate the error between the true and predicted output and multiply it by the learning rate.

        Args:
            y_true              : float - true output
            y_pred              : float - predicted output

        Returns:
            error               : float - error (learn_rate * (y_true - y_pred))
        '''
        return self.learning_rate * (y_true - y_pred)
    
    def update_weights(self, x: np.ndarray, error: float) -> None:
        '''
        Update the weights (w_1, w_2, ..., w_n) of the perceptron by adding the error multiplied by the observation.

        Args:
            x                   : np.ndarray - a single observation with shape (n, )
            error               : float - error (learn_rate * (y_true - y_pred))

        Returns:
            None
        '''
        self.w += error * x # The derivative of the error w.r.t. the weights is (y_true - y_pred) * x
    
    def update_bias(self, error: float) -> None:
        '''
        Update the bias of the perceptron by adding the error.

        Args:
            error               : float - error (y_true - y_pred)

        Returns:
            None
        '''
        self.b += error # The derivative of the error w.r.t. the bias is (y_true - y_pred)
    
    def train(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
        '''
        Train the perceptron by updating the weights and bias for each observation in the dataset.

        Args:
            X                   : np.ndarray - a dataset with shape (m, n)
            Y                   : np.ndarray - a vector of true outputs with shape (m, )

        Returns:
            w                   : np.ndarray - final weights with shape (n, )
            b                   : float - final bias
        '''
        for _ in range(self.n_iterations):
            total_error = 0
            for i in range(len(X)):
                # STEP 1: Predict the output for the current observation
                y_pred = self.predict(X[i])
                # STEP 2: Calculate the error between the true and predicted output
                error = self.calculate_error(Y[i], y_pred)
                # STEP 3: Update the weights and bias
                self.update_weights(X[i], error)
                self.update_bias(error)
                # STEP 4: Accumulate the total error
                total_error += error
            print(f"Epoch #{_ + 1} - Total Error: {total_error}")
        return self.w, self.b
    
# Little demo for AND gate problem - 00, 01, 10, 11 -> 0, 0, 0, 1
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 0, 0, 1])

# Create zero weights and bias
w = np.zeros(2)
b = 0
learning_rate = 0.1
epochs = 3

perceptron = Perceptron(learning_rate=learning_rate, n_iterations=epochs)
w, b = perceptron.train(X, Y)

print(f"Final weights: {w} and final bias: {b}")