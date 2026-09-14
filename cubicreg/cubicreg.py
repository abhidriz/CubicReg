import pandas as pd
import matplotlib.pyplot as plt

dataFrame = pd.read_csv("data/temperature_enzyme_activity.csv")

# Normalization to help adjust values from a 0-1 scale
# Uses 0-1 scaling

x_raw = dataFrame.iloc[:, 0]
y_raw = dataFrame.iloc[:, 1]

x_min, x_max = x_raw.min(), x_raw.max()
y_min, y_max = y_raw.min(), y_raw.max()

x_scaled = (x_raw - x_min) / (x_max - x_min)
y_scaled = (y_raw - y_min) / (y_max - y_min)

# This is the gradient descent function
# It takes the current coefficients (a, b, c, d), the dataset, and the learning rate (alpha)
# It returns updated coefficients after one iteration

def grad_descent_cubic(a, b, c, d, x_data, y_data, alpha):
    a_gradient, b_gradient, c_gradient, d_gradient = 0, 0, 0, 0
    n = len(x_data)
    for i in range(n):

        prediction = a * (x_data[i] ** 3) + b * (x_data[i] ** 2) + c * x_data[i] + d
        error = y_data[i] - prediction

        # Gradients that are changed through iteration
        a_gradient += -2 / n * (x_data[i] ** 3) * error
        b_gradient += -2 / n * (x_data[i] ** 2) * error
        c_gradient += -2 / n * x_data[i] * error
        d_gradient += -2 / n * error

    # Update rule in gradient desc
    a_after = a - alpha * a_gradient
    b_after = b - alpha * b_gradient
    c_after = c - alpha * c_gradient
    d_after = d - alpha * d_gradient

    # Apply the update rule: new = old - (alpha * gradient)
    return a_after, b_after, c_after, d_after

# starting weights
# learning rate at which it assists the model
# epochs are how many rounds the code goes through the dataset
a, b, c, d = 0, 0, 0, 0
alpha = 0.1
epochs = 2000

for i in range(epochs):
    a, b, c, d = grad_descent_cubic(a, b, c, d, x_scaled, y_scaled, alpha)

# Scaling that is used for plotting right after
y_preds_scaled = [a * (val ** 3) + b * (val ** 2) + c * val + d for val in x_scaled]
y_preds_final = [(val * (y_max - y_min)) + y_min for val in y_preds_scaled]

plt.scatter(x_raw, y_raw, color="black", label="Radiation Data")
plt.plot(x_raw, y_preds_final, color="red", label="Cubic Best Fit")
plt.legend()
# plt.show() <--- The show method only works on non server environment. Since I am on rlogin I have to use
plt.savefig("tests/cubic_regression_output.png")
