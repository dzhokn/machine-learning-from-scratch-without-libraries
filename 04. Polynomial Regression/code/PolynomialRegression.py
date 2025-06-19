import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression:
    '''
    Polynomial Regression model. It is a type of regression analysis in which the relationship between $x$ and $y$ 
    is modeled as an nth-degree polynomial. It is an extension of linear regression, but capturing non-linear 
    relationships between $x$ and $y$.

    It supports only 2nd degree polynomials for now.
    '''
    learn_rate: float
    epochs: int
    weights: np.ndarray
    losses: list[float]

    # Initialize the model
    def __init__(self, learn_rate: float, epochs: int, num_features: int):
        '''
        Initializes the Polynomial Regression model.

        Args:
            degree          : int - degree of the polynomial
            learn_rate      : float - learning rate
            epochs          : int - number of epochs
        '''
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.initialize_weights(num_features)
        self.losses = []

    def initialize_weights(self, num_features: int):
        '''
        Initializes the weights of the polynomial regression model. The number of weights depend on 
        the number of features and the degree of the polynomial.

        Args:
            num_features        : int - number of features

        Returns:
            weights             : np.ndarray - weights of the polynomial regression model
        '''
        second_degree_poly_sizes = {
            1: 3,
            2: 6,
            3: 10,
            4: 15,
            5: 21,
            6: 28 # The formula is [1 + num_features + num_features + (num_features choose 2)]
        }
        # Determine the number of weights based on the number of features.
        weight_size = second_degree_poly_sizes[num_features]
        # Initialize the weights with random values.
        self.weights = np.random.randn(weight_size)

    def transform_to_2nd_degree_polynomial(self, x: np.ndarray) -> list[float]:
        '''
        Transforms the input vector of feature values (x_1, x_2, ..., x_n) to a polynomial vector of degree 2.

        Args:
            x               : np.ndarray of shape (n, 1), where n is the number of features

        Returns:
            terms           : list (1, x_1, x_2, ..., x_n, x_1*x_2, x_1*x_3, ..., x_n-1*x_n, x_1^2, x_2^2, ..., x_n^2)
        '''
        n = x.shape[0] # Number of features
        # STEP 1: We start with 1, because the polynomial always has 1 as the first term
        terms = [1]

        # STEP 2: Now we add all the first degree terms (x_i)
        for i in range(n):
            terms.append(x[i])

        # STEP 3: Now we add all the combinations of the first degree terms (x_i * x_j).
        for i in range(n):
            for j in range(i+1, n): # We start from i+1 to avoid duplicates
                terms.append(x[i] * x[j])

        # STEP 4: Now we add all the second degree terms (x_i^2)
        for i in range(n):
            terms.append(x[i]**2)

        return terms

    def transform_to_2nd_degree_matrix(self, X: np.ndarray) -> np.ndarray:
        '''
        Transforms the input matrix of feature vectors (X_1, X_2, ..., X_n) to a polynomial matrix of degree 2
        (1, x_1, x_2, ..., x_n, x_1*x_2, x_1*x_3, ..., x_n-1*x_n, x_1^2, x_2^2, ..., x_n^2).

        Args:
            X               : np.ndarray of shape (m, n), where m is the number of observations and n is the number of features

        Returns:
            output_matrix   : np.ndarray of shape (m, 1 + n + n + (n choose 2) + n)
        '''
        output_matrix = []
        for x in X:
            X_poly = self.transform_to_2nd_degree_polynomial(x)
            output_matrix.append(X_poly)
        return np.array(output_matrix)

    def train_iteration(self, X: np.ndarray, Y: np.ndarray):
        '''
        Trains the polynomial regression model for a single iteration using the gradient descent algorithm.

        Args:
            X               : np.ndarray of shape (m, n), where m is the number of observations and n is the number of features
            Y               : np.ndarray of shape (m, 1), where m is the number of observations
        '''
        # Transform the flat observations matrix to a polynomial matrix
        X_poly = self.transform_to_2nd_degree_matrix(X) # (m, 6)
        # Calculate the predicted values
        Y_pred = np.dot(X_poly, self.weights) # (m,6) @ (6,1) = (m, 1)
        # Calculate the error
        error = Y_pred - Y # (m, 1) - (m, 1) = (m, 1)
        # Update the weights
        self.weights -= self.learn_rate * (X_poly.T @ error) # (6,m) @ (m,1) = (6,1)
        # Update the losses history
        self.losses.append(np.sum(error**2)/X.shape[0])
        
    def train(self, X: np.ndarray, Y: np.ndarray):
        '''
        Trains the polynomial regression model.

        Args:
            X               : np.ndarray of shape (m, n), where m is the number of observations and n is the number of features
            Y               : np.ndarray of shape (m, 1), where m is the number of observations
        '''
        for _ in range(self.epochs):
            self.train_iteration(X, Y)
            # Once in a 100 iterations, print the latest loss
            if _ % 100 == 0:
                print(f"Iteration {_} - Loss: {self.losses[-1]}")

def demonstrate_polynomial_regression():
    '''
    Demonstrates the polynomial regression model training over a sample dataset (salaries in the ML industry).
    '''
    # X1 = years of experience
    X1 = [1.2, 1.3, 1.5, 1.8, 2, 2.1, 2.2, 2.5, 2.8, 2.9, 3.1, 3.3, 3.5, 3.8, 4, 4.1, 4.5, 4.9, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10, 11, 12, 13, 14, 15]
    # X2 = level of education 
    X2 = [2, 5, 3, 5, 3, 4, 2, 3, 4, 4, 3, 7, 5, 6, 5, 5, 2, 3, 4, 5, 6, 7, 5, 3, 2, 4, 5, 7, 3, 5, 7, 7, 5]
    # Y = salary
    Y = [2900, 3300, 3100, 4200, 3500, 3800, 3300, 3500, 3750, 4000, 3900, 5300, 4420, 5000, 4900, 5200, 3900, 4800, 5700, 6500, 6930, 7500, 7360, 6970, 6800, 7500, 8000, 9500, 11000, 9500, 12300, 13700, 12500]
    Y = np.array(Y) / 1000 # Normalize the Y values, so we prevent overflows

    # Merge the X1 and X2 into a single X
    X = [[x1, x2] for x1, x2 in zip(X1, X2)]

    # Convert the X, Y to a numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    # Configure gradient descent settings
    learn_rate = 0.000001   # learning rate (step size)
    iterations = 500   # number of iterations (epochs)

    model = PolynomialRegression(learn_rate=learn_rate, epochs=iterations, num_features=len(X[0]))
    model.train(X, Y)

    print(f"\nFinal weights:")
    print(f"\tb_0    : {model.weights[0]:.2f}")
    print(f"\tx_1    : {model.weights[1]:.2f}")
    print(f"\tx_2    : {model.weights[2]:.2f}")
    print(f"\tx_1*x_2: {model.weights[3]:.2f}")
    print(f"\tx_1^2  : {model.weights[4]:.2f}")
    print(f"\tx_2^2  : {model.weights[5]:.2f}")

    # Plot the loss function
    epochs = np.arange(iterations)
    plt.plot(epochs, model.losses)
    plt.show()

demonstrate_polynomial_regression()