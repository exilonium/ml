import numpy as np


class LogisticLearning:
    def __init__(self, learning_rate=0.1, epochs=1_000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.w = 0.0
        self.b = 0.0

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def predict(self, X):
        z = self.w * X + self.b
        return self.sigmoid(z)

    def classify(self, x):
        return int(self.predict(x) >= 0.5)

    def train(self, X, y):
        for epoch in range(self.epochs):
            # Forward pass
            p = self.predict(X)

            # Loss
            loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

            # Gradients
            error = p - y

            dw = np.mean(error * X)
            db = np.mean(error)

            # Update
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if epoch % 1000 == 0:
                print(f"epoch={epoch}: loss={loss:.6f}, w={self.w:.6f}, b={self.b:.6f}")


X = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([0, 0, 0, 1, 1, 1], dtype=float)


model = LogisticLearning(learning_rate=0.1, epochs=10_000_00)

model.train(X, y)


while True:
    x = float(input("Enter a number (67 to exit): "))

    if x == 67:
        break

    print(f"prediction={model.predict(x):.6f}, classification={model.classify(x)}")
