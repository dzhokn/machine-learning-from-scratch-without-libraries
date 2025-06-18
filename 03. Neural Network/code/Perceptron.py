import numpy as np

class Perceptron:
    learning_rate: float
    n_iterations: int
    w: np.ndarray
    b: float

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = np.zeros(2)
        self.b = 0

    def weighted_sum(self, X: np.ndarray) -> float:
        return np.dot(X, self.w) + self.b
    
    def predict(self, X: np.ndarray) -> float:
        H = self.weighted_sum(X)
        return np.where(H >= 0.5, 1, 0) # Binary classification (0 or 1)
    
    def calculate_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.learning_rate * (y_true - y_pred)
    
    def update_weights(self, X: np.ndarray, error: float) -> None:
        self.w += error * X
    
    def update_bias(self, error: float) -> None:
        self.b += error
    
    def train(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
        for _ in range(self.n_iterations):
            total_error = 0
            for i in range(len(X)):
                y_pred = self.predict(X[i])
                error = self.calculate_error(Y[i], y_pred)
                self.update_weights(X[i], error)
                self.update_bias(error)
            total_error += error
            print(f"Epoch #{_ + 1} - Total Error: {total_error}")
        return self.w, self.b
    
# Little demo
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