import numpy as np
import matplotlib.pyplot as plt


# Let's name the years of experience 'X' (an input variable)
X = [1.2, 1.3, 1.5, 1.8, 2, 2.1, 2.2, 2.5, 2.8, 2.9, 3.1, 3.3, 3.5, 3.8, 4, 4.1, 4.5, 4.9, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10, 11, 12, 13, 14, 15]

# And let's name the corresponding salaries 'Y' (the target/output variable)
Y = [2900, 3300, 3100, 4200, 3500, 3800, 3300, 3500, 3750, 4000, 3900, 5300, 4420, 5000, 4900, 5200, 3900, 4800, 5700, 6500, 6930, 7500, 7360, 6970, 6800, 7500, 8000, 9500, 11000, 9500, 12300, 13700, 12500]

X = np.vectorize(float)(X)
plt.plot(X, 2000 + 800 * X, color='g')
plt.scatter(X, Y)
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

def compute_cost(X: np.ndarray, Y: np.ndarray, w: float, b: float) -> float:
    m = len(X)
    cost = 0
    for i in range(m):
        cost += (w * X[i] + b - Y[i]) ** 2
    return cost / m

print(compute_cost(X, Y, 2000, 800))