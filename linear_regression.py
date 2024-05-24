import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_data():
    """Load and return the example dataset."""
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
    return X, y

def preprocess_data(X):
    """Scale features and add intercept term."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_intercept = np.hstack((np.ones((X_scaled.shape[0], 1)), X_scaled))
    return X_intercept, scaler

def compute_theta_normal_equation(X, y):
    """Compute theta using the normal equation."""
    return np.linalg.inv(X.T @ X) @ X.T @ y

def predict(X, theta):
    """Make predictions using the linear model."""
    return X @ theta

def compute_mse(y_true, y_pred):
    """Compute Mean Squared Error."""
    return mean_squared_error(y_true, y_pred)

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

def plot_cost_history(cost_history, filename='cost_convergence.png'):
    """Plot and save the cost function history."""
    plt.plot(range(len(cost_history)), cost_history, 'b.')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function Convergence')
    plt.savefig(filename)

def main():
    # Load and preprocess data
    X, y = load_data()
    X, scaler = preprocess_data(X)

    # Compute theta using the normal equation
    theta_normal_eq = compute_theta_normal_equation(X, y)

    # Predict using the normal equation
    X_new = np.array([[2500, 4, 2]])
    X_new_scaled = scaler.transform(X_new)
    X_new_intercept = np.hstack((np.ones((X_new_scaled.shape[0], 1)), X_new_scaled))
    y_pred = predict(X_new_intercept, theta_normal_eq)
    print("Predicted price for new data:", y_pred)

    # Predictions on training set
    y_train_pred = predict(X, theta_normal_eq)

    # Compute and print MSE
    mse = compute_mse(y, y_train_pred)
    print(f"Mean Squared Error: {mse}")

    # Perform gradient descent
    initial_theta = np.zeros((X.shape[1], 1))
    learning_rate = 0.001
    iterations = 1000
    theta_gd, cost_history = gradient_descent(X, y, initial_theta, learning_rate, iterations)

    # Plot and save cost history
    plot_cost_history(cost_history)

if __name__ == "__main__":
    main()
