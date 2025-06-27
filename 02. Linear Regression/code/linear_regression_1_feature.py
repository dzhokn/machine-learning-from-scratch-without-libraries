import numpy as np
import matplotlib.pyplot as plt

def compute_gradient(X: np.ndarray, Y: np.ndarray, w: float, b: float) -> tuple[float, float]:
    '''
    Calculate the derivatives of the cost function with respect to each weight and bias.

    Returns:
        dj_dw -> float          : derivative of the cost function with respect to the weight
        dj_db -> float          : derivative of the cost function with respect to the bias
    '''
    m = len(X) # Number of observations

    # Initialize the derivatives
    dj_db = 0 # Initialize the derivative of the cost function with respect to the bias
    dj_dw = 0 # Initialize the derivative of the cost function with respect to the weight
    total_cost = 0 # Initialize the total cost

    for i in range(m): # For each observation
        y_pred_i = X[i] * w + b     # Calculate the predicted value for the current observation
        cost_i = y_pred_i - Y[i]    # Calculate the cost for the current observation
        dj_dw += cost_i * X[i]      # Calculate the derivative of the cost function with respect to the weight
        dj_db += cost_i             # Calculate the derivative of the cost function with respect to the bias
        total_cost += cost_i ** 2   # Calculate the total cost

    # Normalize the derivatives
    dj_dw = 2/m * dj_dw # Calculate the derivative of the cost function with respect to the weight
    dj_db = 2/m * dj_db # Calculate the derivative of the cost function with respect to the bias

    # Return the derivatives
    return dj_dw, dj_db, total_cost

def fit(X: list[list[float]], Y: list[float], learn_rate: float, epochs: int) -> tuple[float, float]:
    '''
    Perform gradient descent to find the optimal weights and bias.

    Args:
        X -> list[float]    : m rows of 1 feature (x^1, x^2, ..., x^m)
        Y -> list[float]    : m rows of expected float values (y^1, y^2, ..., y^m)
        learn_rate -> float : learning rate
        epochs -> int       : number of iterations to perform

    Returns:
        weights -> float    : weight
        bias -> float       : bias
    '''
    # Initialize the weights and bias
    weight = 0
    bias = 0

    # Perform gradient descent
    for epoch_number in range(epochs):
        # Calculate the derivative of the cost function with respect to the weights and bias
        dj_dw, dj_db, total_cost = compute_gradient(X, Y, weight, bias) 

        # Update the weights and bias
        weight = weight - learn_rate * dj_dw
        bias = bias - learn_rate * dj_db

        # Print the cost every 100 iterations
        if epoch_number % 1000 == 0:
            print(f"Epoch {epoch_number}: Cost = {total_cost}")
    return weight, bias



# Let's name the years of experience 'X' (an input variable)
X = np.array([1.2, 1.3, 1.5, 1.8, 2, 2.1, 2.2, 2.5, 2.8, 2.9, 3.1, 3.3, 3.5, 3.8, 4, 4.1, 4.5, 4.9, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10, 11, 12, 13, 14, 15])

# And let's name the corresponding salaries 'Y' (the target/output variable)
Y = np.array([2900, 3300, 3100, 4200, 3500, 3800, 3300, 3500, 3750, 4000, 3900, 5300, 4420, 5000, 4900, 5200, 3900, 4800, 5700, 6500, 6930, 7500, 7360, 6970, 6800, 7500, 8000, 9500, 11000, 9500, 12300, 13700, 12500])

learn_rate = 0.01
epochs = 5000
weight, bias = fit(X, Y, learn_rate, epochs)

# Print the results
print(f"The final linear equation is: y = {weight:.2f}x + {bias:.2f}")

# Plot the results
plt.scatter(X, Y, color='blue', label='Actual')
plt.plot(X, weight * X + bias, color='r', label='Predicted')
plt.legend()
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()