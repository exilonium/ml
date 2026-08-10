# just a fun linear reg learning to predict the sgpa

import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.0001, epochs=1_000_000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = 100
        self.b = 0.0

    def predict(self, X):
        return self.w * X + self.b

    def train(self, X, y):
        for epoch in range(self.epochs):
            prediction = self.predict(X)

            error = prediction - y

            loss = np.mean(error**2)

            dw = np.mean(2 * error * X)
            db = np.mean(2 * error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if epoch % 10000 == 0:
                print(f"Epoch={epoch}: loss={loss:.4f}, w={self.w:.4f}, b={self.b:.4f}")

    def __repr__(self) -> str:
        return f"LinearRegression(w={self.w:.4f}, b={self.b:.4f})"


# Data

X = np.array([1, 2, 3, 4])

y = np.array([6.64, 6.64, 7.00, 7.04])  # exil 7.22

aks = np.array([7.68, 7.73, 7.44, 7.35])  # akshansh 7.23


# Creating model

model = LinearRegression(learning_rate=0.0001, epochs=1_000_000)


# Train

model.train(X, aks)


print("\n\nFinal:\n")

print(f"w = {model.w}")
print(f"b = {model.b}")


# Predict 5th semester

semester = 5
prediction = model.predict(semester)

print(f"You will get {prediction:.2f} SGPA in sem#{semester}")
