import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Example data
X = np.array([[2104, 5, 1], 
              [1600, 3, 2], 
              [2400, 3, 2], 
              [1416, 2, 1], 
              [3000, 4, 3]])

y = np.array([[399900], 
              [329900], 
              [369000], 
              [232000], 
              [539900]])

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add intercept term to X
m, n = X.shape
X = np.hstack((np.ones((m, 1)), X))

# Compute theta using the normal equation
theta = np.linalg.inv(X.T @ X) @ X.T @ y

# New data example
X_new = np.array([[2500, 4, 2]])
X_new = scaler.transform(X_new)

# Add intercept term to new data
X_new = np.hstack((np.ones((X_new.shape[0], 1)), X_new))

# Make prediction
y_pred = X_new @ theta
print("Predicted price for new data:", y_pred)

# Predictions on training set
y_train_pred = X @ theta

# Compute MSE
mse = mean_squared_error(y, y_train_pred)
print(f"Mean Squared Error: {mse}")

def compute_cost(X, y, theta):
    """Compute the cost for linear regression."""
    m = len(y)
    predictions = X @ theta
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    """Perform gradient descent to learn theta."""
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        gradients = (1 / m) * X.T @ (X @ theta - y)
        theta = theta - learning_rate * gradients
        cost_history[i] = compute_cost(X, y, theta)
    
    return theta, cost_history

# Initialize parameters for gradient descent
theta = np.zeros((n + 1, 1))
learning_rate = 0.001  # Reduced learning rate
iterations = 1000

# Perform gradient descent
theta, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)

# Plotting the cost function history
plt.plot(range(iterations), cost_history, 'b.')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.savefig('cost_convergence.png')
