import numpy as np


class LogisticLearning:
    def __init__(self, learning_rate=0.1, epochs=1_000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.w = None
        self.b = 0.0

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def predict(self, X):
        z = X @ self.w + self.b
        return self.sigmoid(z)

    def classify(self, x):
        return int(self.predict(x) >= 0.5)

    def train(self, X, y):
        # One weight for every feature
        self.w = np.zeros(X.shape[1])

        for epoch in range(self.epochs):
            p = self.predict(X)

            # Prevent log(0)
            p = np.clip(p, 1e-15, 1 - 1e-15)

            loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

            error = p - y

            dw = (X.T @ error) / len(X)  # X.T means X transpose
            db = np.mean(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if epoch % 1000 == 0:
                print(f"epoch={epoch}: loss={loss:.6f}, w={self.w}, b={self.b:.6f}")


X = np.array(
    [
        [1, 5, 60],
        [2, 5, 65],
        [2, 6, 70],
        [4, 7, 80],
        [5, 7, 90],
        [6, 8, 95],
    ],
    dtype=float,
)

y = np.array([0, 0, 0, 1, 1, 1], dtype=float)


model = LogisticLearning(learning_rate=0.1, epochs=10_000)

model.train(X, y)


while True:
    x = input("Enter study_hours sleep_hours attendance (or 67 to exit): ")

    if x == "67":
        break

    x = np.array([float(v) for v in x.split()])

    probability = model.predict(x)
    classification = model.classify(x)

    print(f"prediction={probability:.6f}, classification={classification}")
