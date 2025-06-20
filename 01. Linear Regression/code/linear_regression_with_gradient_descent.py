import matplotlib.pyplot as plt

class LinearRegression:
    '''
    Linear Regression model based on gradient descent that uses all the training examples at once to update the weights and bias.
    '''

    def __calculate_y(self, x: list[float]) -> float:
        '''
        Calculate the predicted value for a single observation.

        Args:
            x -> list[float]    : list of features (x1, x2, ..., xn)

        Returns:
            y -> float          : predicted value for a single observation
        '''
        n = len(x) # Number of features
        y = self.bias
        for k in range(n):
            y += self.weights[k] * x[k] # y = b + w1*x1 + w2*x2 + ... + wn*xn
        return y

    def __calculate_cost(self) -> float:
        '''
        Calculate the cost by comparing the predicted and actual values for all observations.

        Returns:
            cost -> float           : cost of the model
        '''
        m = len(self.Y) # Number of observations
        cost = 0
        for i in range(m): # For each observation
            y_pred_i = self.__calculate_y(self.X[i]) # Calculate the predicted value for the current observation
            cost += (y_pred_i - self.Y[i]) ** 2 # Calculate the squared difference between the predicted and actual values
        return cost / m # Normalize by the number of observations

    def __calculate_gradient(self) -> tuple[list[float], float]:
        '''
        Calculate the derivatives of the cost function with respect to each weight and bias.

        Returns:
            dj_dw -> list[float]    : n weight derivatives (dj_dw1, dj_dw2, ..., dj_dwn)
            dj_db -> float          : derivative of the cost function with respect to the bias
        '''
        # Prepare dimensions
        m = len(self.Y) # Number of observations
        n = len(self.weights) # Number of features

        # Initialize the derivatives
        dj_db = 0 # Initialize the derivative of the cost function with respect to the bias
        dj_dw = [0] * n # Initialize the derivative of the cost function with respect to the weights. n-sized list of zeros.

        for i in range(m): # For each observation
            y_pred_i = self.__calculate_y(self.X[i]) # Calculate the predicted value for the current observation
            cost_i = y_pred_i - self.Y[i] # Calculate the cost for the current observation
            for j in range(n): # For each feature
                dj_dw[j] += cost_i * self.X[i][j] # Calculate the derivative of the cost function with respect to the weights
            dj_db += cost_i # Calculate the derivative of the cost function with respect to the bias

        # Normalize the derivatives
        for k in range(n): # For each feature
            dj_dw[k] = 2/m * dj_dw[k] # Calculate the derivative of the cost function with respect to the weights
        dj_db = 2/m * dj_db # Calculate the derivative of the cost function with respect to the bias

        # Return the derivatives
        return dj_dw, dj_db

    def fit(self, X: list[list[float]], Y: list[float], learn_rate: float, num_iterations: int) -> tuple[list[float], float]:
        '''
        Perform gradient descent to find the optimal weights and bias.

        Args:
            X -> list[list[float]]  : m rows of n features each (x^1, x^2, ..., x^m)
            Y -> list[float]        : m rows of expected float values (y^1, y^2, ..., y^m)
            learn_rate -> float     : learning rate
            num_iterations -> int   : number of iterations to perform

        Returns:
            weights -> list[float]  : n weights (w1, w2, ..., wn)
            bias -> float           : bias
        '''
        # Initialize the weights and bias
        self.X = X
        self.Y = Y
        self.weights = [0] * len(X[0])   # Initialize the weights to 0. The number of weights is equal to the number of features.
        self.bias = 0                    # Initialize the bias to 0.
        self.learn_rate = learn_rate
        self.num_iterations = num_iterations

        # Perform gradient descent
        for iteration_number in range(self.num_iterations):
            # Calculate the derivative of the cost function with respect to the weights and bias
            dj_dw, dj_db = self.__calculate_gradient() 

            # Update the weights and bias
            self.weights = [self.weights[j] - self.learn_rate * dj_dw[j] for j in range(len(self.weights))]
            self.bias = self.bias - self.learn_rate * dj_db

            # Print the cost every 100 iterations
            if iteration_number % 100 == 0:
                print(f"Iteration {iteration_number}: Cost = {self.__calculate_cost()}")
        return self.weights, self.bias

class Demo:
    '''
    Demo class to run the batch gradient descent algorithm.
    '''

    def __prepare_data(self) -> tuple[list[list[float]], list[float]]:
        '''
        Prepare the data for the batch gradient descent algorithm.

        Returns:
            X1 -> list[float]    : years of experience
            X2 -> list[float]    : level of education
            Y -> list[float]     : salary
        '''
        # X1 = years of experience
        X1 = [1.2, 1.3, 1.5, 1.8, 2, 2.1, 2.2, 2.5, 2.8, 2.9, 3.1, 3.3, 3.5, 3.8, 4, 4.1, 4.5, 4.9, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10, 11, 12, 13, 14, 15]
        # X2 = level of education 
        X2 = [2, 5, 3, 5, 3, 4, 2, 3, 4, 4, 3, 7, 5, 6, 5, 5, 2, 3, 4, 5, 6, 7, 5, 3, 2, 4, 5, 7, 3, 5, 7, 7, 5]
        # Y = salary
        Y = [2900, 3300, 3100, 4200, 3500, 3800, 3300, 3500, 3750, 4000, 3900, 5300, 4420, 5000, 4900, 5200, 3900, 4800, 5700, 6500, 6930, 7500, 7360, 6970, 6800, 7500, 8000, 9500, 11000, 9500, 12300, 13700, 12500]
        return X1, X2, Y

    def __plot_results(self, X1: list[float], X2: list[float], Y: list[float], weights: list[float], bias: float) -> None:
        '''
        Plot the results of the batch gradient descent algorithm.

        Args:
            X1 -> list[float]    : years of experience
            X2 -> list[float]    : level of education
            Y -> list[float]     : salary
            weights -> list[float]     : weights
            bias -> float           : bias

        Returns:
            None
        '''
        # Calculate the predicted values
        Y_pred = [bias + weights[0] * x1 + weights[1] * x2 for x1, x2 in zip(X1, X2)]

        plt.scatter(X1, Y, color='blue', label='Actual')
        plt.scatter(X1, Y_pred, color='red', label='Predicted')
        plt.legend()
        plt.show()

    def run(self) -> None:
        '''
        Run the batch gradient descent algorithm.

        Returns:
            None
        '''
        X1, X2, Y = self.__prepare_data()

        # Merge the X1 and X2 into a single X
        X = [[x1, x2] for x1, x2 in zip(X1, X2)]

        # Configure gradient descent settings
        learn_rate = 0.01   # learning rate (step size) - usually between 0.001 and 0.1
        iterations = 5000   # number of iterations (epochs)

        # Execute the gradient descent in search for optimal cost function (optimal `w` and `b`)
        model = LinearRegression()
        w, b = model.fit(X, Y, learn_rate, iterations)
        print(f"Final `b` is {b} and `w` are {w}")

        self.__plot_results(X1, X2, Y, w, b)

if __name__ == "__main__":
    demo = Demo()
    demo.run()