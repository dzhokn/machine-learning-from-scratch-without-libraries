import numpy as np

class StochasticGradientDescent:
    ''' Stochastic Gradient Descent is a type of gradient descent that uses a single training example to update the weights and bias.'''

    def __init__(self, X: np.ndarray, Y: np.ndarray, learn_rate: float, num_iterations: int):
        '''
        Initialize the class with the data and the learning rate and the number of iterations

        Args:
            X -> np.ndarray  : m rows of n features each (x^1, x^2, ..., x^m)
            Y -> np.ndarray  : m rows of expected float values (y^1, y^2, ..., y^m)
            learn_rate -> float : learning rate
            num_iterations -> int : number of iterations to perform
        '''
        self.X = X
        self.Y = Y
        self.w = [0] * len(X[0]) # Initialize the weights to 0. The number of weights is equal to the number of features.
        self.b = 0 # Initialize the bias to 0.
        self.learn_rate = learn_rate
        self.num_iterations = num_iterations

    def __calculate_y(self, x: list[float], w: list[float], b: float) -> float:
        '''
        Calculate the predicted value for a single observation.

        Args:
            x -> list[float]    : list of features (x1, x2, ..., xn)
            w -> list[float]    : list of weights (w1, w2, ..., wn)
            b -> float          : bias

        Returns:
            y -> float          : predicted value for a single observation
        '''
        n = len(x) # Number of features
        y = b
        for k in range(n):
            y += w[k] * x[k] # y = w1*x1 + w2*x2 + ... + wn*xn + b
        return y
    
    def __calculate_cost(self, y: float, x: list[float], w: list[float], b: float) -> float:
        '''
        Calculate the cost by comparing the predicted and actual value for SINGLE observation.

        Args:
            y -> float              : expected float value (y)
            x -> list[float]        : n features (x1, x2, ..., xn)
            w -> list[float]        : n weights (w1, w2, ..., wn)
            b -> float              : bias

        Returns:
            cost -> float           : cost for this observation
        '''
        y_pred = self.__calculate_y(x, w, b) # Calculate the predicted value for the this observation
        return (y_pred - y) ** 2 # Calculate the squared difference between the predicted and actual value
    
    def __calculate_dj_dw(self, y: float, x: list[float], w: list[float], b: float) -> list[float]:
        '''
        Calculate the derivative of the cost function with respect to each weight.

        Args:
            y -> float              : expected float value (y)
            x -> list[float]        : n features (x1, x2, ..., xn)
            w -> list[float]        : n weights (w1, w2, ..., wn)
            b -> float              : bias

        Returns:
            dj_dw -> list[float]    : n weight derivatives (dj_dw1, dj_dw2, ..., dj_dwn)
        '''
        n = len(w) # Number of features
        dj_dw = [0] * n # Initialize the derivative of the cost function with respect to the weights. n-sized list of zeros.

        y_pred = self.__calculate_y(x, w, b) # Calculate the predicted value for the current observation
        for j in range(n): # For each feature
            dj_dw[j] += 2 * (y_pred - y) * x[j] # Calculate the derivative of the cost function with respect to the weights

        return dj_dw

    def __calculate_dj_db(self, y: float, x: list[float], w: list[float], b: float) -> float:
        '''
        Calculate the derivative of the cost function with respect to the bias.

        Args:
            y -> float              : expected float value (y)
            x -> list[float]        : n features (x1, x2, ..., xn)
            w -> list[float]        : n weights (w1, w2, ..., wn)
            b -> float              : bias

        Returns:
            dj_db -> float          : derivative of the cost function with respect to the bias
        '''
        y_pred = self.__calculate_y(x, w, b) # Calculate the predicted value for the current observation
        dj_db = 2 * (y_pred - y) # Calculate the derivative of the cost function with respect to the bias
        return dj_db
    
    def run(self) -> tuple[list[float], float]:
        '''
        Perform stochasticgradient descent to find the optimal weights and bias.

        Returns:
            w -> list[float]        : n weights (w1, w2, ..., wn)
            b -> float              : bias
        '''
        for iteration_number in range(self.num_iterations):

            # Randomize the order of the observations
            m = len(self.X)
            random_indices = np.random.permutation(m)
            X_shuffled = self.X[random_indices]
            Y_shuffled = self.Y[random_indices]

            for i in range(m):
                x = X_shuffled[i] # Get a single observation
                y = Y_shuffled[i] # Get the expected value for that observation

                dj_dw = self.__calculate_dj_dw(y, x, self.w, self.b) # Calculate the derivative of the cost function with respect to the weights
                dj_db = self.__calculate_dj_db(y, x, self.w, self.b) # Calculate the derivative of the cost function with respect to the bias

                # Update the weights and bias
                self.w = [self.w[j] - self.learn_rate * dj_dw[j] for j in range(len(self.w))]
                self.b = self.b - self.learn_rate * dj_db

            # Print the cost every 100 iterations
            if iteration_number % 1000 == 0:
                print(f"Iteration {iteration_number}: Total cost = {self.__calculate_total_cost()}")
        return self.w, self.b

    def __calculate_total_cost(self) -> float:
        '''Calculate the total cost by comparing the predicted and actual values for ALL observations.'''
        m = len(self.Y) # Number of observations
        cost = 0
        for i in range(m):                      # For each observation
            y_pred_i = self.X[i] @ self.w + self.b             # Calculate the predicted value for the current observation
            cost += (y_pred_i - self.Y[i]) ** 2      # Calculate the squared difference between the predicted and actual values
        return cost / m                         # Normalize by the number of observations
    
class Demo:
    '''
    Demo class to run the stochastic gradient descent algorithm.
    '''

    def __prepare_data(self) -> tuple[np.ndarray, np.ndarray]:
        '''
        Prepare the data for the stochastic gradient descent algorithm.
        '''
        # X1 = years of experience
        X1 = [1.2, 1.3, 1.5, 1.8, 2, 2.1, 2.2, 2.5, 2.8, 2.9, 3.1, 3.3, 3.5, 3.8, 4, 4.1, 4.5, 4.9, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10, 11, 12, 13, 14, 15]
        # X2 = level of education 
        X2 = [2, 5, 3, 5, 3, 4, 2, 3, 4, 4, 3, 7, 5, 6, 5, 5, 2, 3, 4, 5, 6, 7, 5, 3, 2, 4, 5, 7, 3, 5, 7, 7, 5]
        # Y = salary
        Y = [2900, 3300, 3100, 4200, 3500, 3800, 3300, 3500, 3750, 4000, 3900, 5300, 4420, 5000, 4900, 5200, 3900, 4800, 5700, 6500, 6930, 7500, 7360, 6970, 6800, 7500, 8000, 9500, 11000, 9500, 12300, 13700, 12500]
        
        # Merge the X1 and X2 into a single X
        X = [[x1, x2] for x1, x2 in zip(X1, X2)]

        # Convert the X, Y to a numpy arrays
        X = np.array(X)
        Y = np.array(Y)

        return X, Y
    
    def run(self) -> None:
        '''
        Run the batch gradient descent algorithm.

        Returns:
            None
        '''
        X, Y = self.__prepare_data()

        learn_rate = 0.00001   # learning rate (step size) - usually between 0.001 and 0.1
        iterations = 100000   # number of iterations (epochs)

        # Execute the gradient descent in search for optimal cost function (optimal `w` and `b`)
        sgd = StochasticGradientDescent(X, Y, learn_rate, iterations)
        w, b = sgd.run()
        print(f"Final `b` is {b} and `w` are {w}")

if __name__ == "__main__":
    demo = Demo()
    demo.run()