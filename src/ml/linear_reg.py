import numpy as np

# Training data
X = np.array([10, 20, 30, 40, 50], dtype=float)
y = np.array([200, 400, 600, 800, 1000], dtype=float)

# Model parameters
w = 0.0
b = 0.0

# Hyperparameters
learning_rate = 0.0001
epochs = 300000

# Gradient descent
for epoch in range(epochs):
    # Prediction
    prediction = w * X + b

    # Error
    error = prediction - y

    # Mean squared error
    loss = np.mean(error**2)

    # Gradients
    dw = np.mean(2 * error * X)
    db = np.mean(2 * error)

    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

# Final parameters
print("\nFinal:")
print(f"w = {w}")
print(f"b = {b}")

# Prediction for x = 60
x = 100
prediction = w * x + b

print(f"Prediction for {x} = {prediction}")
