# Linear Regression Analysis

This project demonstrates the application of linear regression to predict housing prices based on multiple features. It utilizes Python libraries such as NumPy, scikit-learn, and Matplotlib to manage data, compute predictions, and visualize results.

## Dependencies

- Python 3.x
- NumPy
- scikit-learn
- Matplotlib

## Setup

To run this project, you need to install the required Python libraries. You can install these packages via pip:

```bash
pip install numpy matplotlib scikit-learn
```

## Usage

The script can be executed directly if named main.py:
```bash
python linear_regression.py
```

This will run the main function that includes data loading, preprocessing, model training using the normal equation and gradient descent, and plotting of the cost function over iterations.
Features

    **Data Loading**: Load a simple predefined dataset.
    Data Preprocessing: Scale features using StandardScaler and add an intercept term.
    Model Training: Utilize the normal equation for an analytical solution to the linear regression.
    Prediction: Predict housing prices based on input features.
    Cost Computation: Compute the mean squared error and the regression cost.
    Gradient Descent: An iterative method for optimizing the model's parameters.
    Visualization: Plot the convergence of the cost function during training.

## Functions
load_data()

Load and return the housing price dataset.
preprocess_data(X)

Scale features and add an intercept term to the dataset.
compute_theta_normal_equation(X, y)

Compute and return the optimal weights using the normal equation.
predict(X, theta)

Make predictions using the learned weights on the input data.
compute_mse(y_true, y_pred)

Compute and return the mean squared error between true and predicted values.
compute_cost(X, y, theta)

Compute the cost for the current set of parameters.
gradient_descent(X, y, theta, learning_rate, iterations)

Perform gradient descent to learn the model parameters, given initial conditions and hyperparameters.
````plot_cost_history(cost_history, filename)```

Plot and save the cost function history to a file.


## Warning Configuration

Warnings are suppressed in this script using the warnings library to avoid clutter during demonstration. However, for development, it is often better to enable warnings.
```bash
import warnings
warnings.filterwarnings("ignore")
```


## Conclusion

This script provides a basic framework for implementing linear regression and can be expanded with additional features, larger datasets, or different optimization algorithms.

## License

This project is free to use and modify for educational purposes.
