import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    Logistic Regression model
    """

    # Model parameters
    learn_rate: float
    num_epochs: int
    weights: np.ndarray
    bias: float

    # Constructor
    def __init__(self, learn_rate: float, num_epochs: int):
        """
        Initialize the model parameters

        Args:
            learn_rate (float)          : The learning rate
            num_epochs (int)            : The number of epochs
        """
        self.learn_rate = learn_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None
    
    def compute_y(self, x: np.ndarray) -> float:
        """
        Compute the sigmoid of z

        Args:
            x (np.ndarray)              : A vector (observation)

        Returns:
            float                       : sigmoid(z)
        """
        z = self.__compute_z(x)
        return self.__sigmoid(z)

    def __sigmoid(self, z: float) -> float:
        return 1 / (1 + np.exp(-z)) # Sigmoid function which returns a value between 0 and 1

    def __compute_z(self, x: np.ndarray) -> float:
        return self.bias + self.weights @ x # (1, n) * (n, 1) = (1,) , where n is the number of features
    
    def compute_loss(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculates the loss (cost) for a given X (set of vectors), Y (target values).

        Args:
            X (np.ndarray)              : The input data (m, n)
            Y (np.ndarray)              : The target values (m, 1)

        Returns:
            float                       : The loss
        """ 
        m = X.shape[0]        # The number of examples (rows) in the training set
        loss = 0.

        for i in range(m):
            y_pred = self.compute_y(X[i])
            loss += -(Y[i] * np.log(y_pred) + (1-Y[i]) * np.log(1-y_pred)) # Cross-Entropy
        return loss / m
    
    def compute_gradient(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]: 
        """
        Computes the gradient for logistic regression with multiple features.

        Args:
            X (ndarray (m,n))           : input data, m rows with n features
            Y (ndarray (m,))            : target values
        
        Returns:
            dj_dw (ndarray (n,))        : The gradient of the cost w.r.t. the parameters w. 
            dj_db (scalar)              : The gradient of the cost w.r.t. the parameter b. 
        """
        # Calculate all the predicted values
        m = X.shape[0]               # number of rows
        Y_pred = np.zeros(m)
        for i in range(m):
            Y_pred[i] = self.compute_y(X[i])

        # Calculate the gradient
        dj_dw = X.T @ (Y_pred - Y) / m
        dj_db = np.sum(Y_pred - Y) / m

        return dj_db, dj_dw
    
    def fit(self, X: np.ndarray, Y: np.ndarray): 
        """
        Performs batch gradient descent to learn w and b. Updates w and b by taking 
        num_iters gradient steps with specified learning rate.

        Args:
        X (ndarray (m,n))       : input data, m examples with n features
        Y (ndarray (m,))        : target values

        Returns:
        weights (ndarray (n,))  : Updated values of parameters 
        bias (float)            : Updated value of parameter 
        losses (list)           : History of losses
        """
        losses = []            # An array to store history of losses

        # Initialize the weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.num_epochs):
            # Calculate the gradient for these `w` and `b`
            dj_db, dj_dw = self.compute_gradient(X, Y)

            # Update `w` and `b` based on the gradient (up or down) and the specified learning rate
            self.weights -= self.learn_rate * dj_dw   # vector operation (w and dj_dw are vectors with the same size)
            self.bias -= self.learn_rate * dj_db

            # Save loss at each iteration
            losses.append( self.compute_loss(X, Y) )

            # Print the cost after every 10% of the iterations
            if _ % 1000 == 0:
                print(f"Iteration {_} - Loss: {losses[-1]:.4f}")

        return self.weights, self.bias, losses # return final w,b and J history for graphing
    
    def predict(self, x: np.ndarray) -> float:
        '''
        Predict the probability of a binary outcome based on the input features.

        Args:
        x (ndarray): A vector (observation)

        Returns:
        y_pred (int): 1 if sigmoid(z) > 0.5 else 0
        '''
        y_pred = self.compute_y(x)
        return 1 if y_pred > 0.5 else 0
    
    def compute_accuracy(self, X: np.ndarray, Y: np.ndarray) -> float:
        '''
        Compute the accuracy of the model.

        Args:
        X (ndarray): A dataset of input features
        Y (ndarray): A dataset of target values
        '''
        m = X.shape[0]
        correct_predictions = 0
        for i in range(m):
            y_pred = self.predict(X[i])
            if y_pred == Y[i]:
                correct_predictions += 1
        return correct_predictions / m
    
def demo():
    # X1 = Basketball players - years of experience
    X1 = [1.2, 1.3, 1.5, 1.8, 2, 2.1, 2.2, 2.5, 2.8, 2.9, 3.1, 3.3, 3.5, 3.8, 4, 4.1, 4.5, 4.9, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10, 11, 12, 13, 14, 15]
    # X2 = Height (cm)
    X2 = [181, 190, 202, 185, 178, 189, 204, 192, 183, 184, 195, 196, 197, 188, 189, 190, 191, 182, 183, 190, 185, 186, 197, 191, 189, 200, 186, 189, 191, 190, 195, 193, 187, 192, 190, 187]
    # Y = Win or Lose
    Y = [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # Merge the X1 and X2 into a single X
    X = [[x1, x2] for x1, x2 in zip(X1, X2)]

    # Convert the X, Y to a numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    # Configure gradient descent settings
    learn_rate = 0.0002   # learning rate (step size)
    epochs = 20000   # number of iterations (epochs)

    model = LogisticRegression(learn_rate, epochs)
    weights, bias, losses = model.fit(X, Y)

    print(f"\nFinal weights: {weights} and bias: {bias}")
    print(f"Accuracy: {model.compute_accuracy(X, Y) * 100:.0f}%")

    # Plot the loss function
    plt.plot(np.arange(epochs), losses)
    plt.show()

demo()