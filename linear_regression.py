import sys
import numpy as np


class LinearRegression:
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, Y):
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float).reshape(-1)

        n_features = self.X.shape[1]
        self.W = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.iterations):
            self.update_weights()

        return self

    def update_weights(self):
        # Ошибка прогноза для каждого объекта
        error = self.predict(self.X) - self.Y

        # Градиенты среднеквадратичной ошибки по весам и смещению
        n_samples = self.X.shape[0]
        dW = (2.0 / n_samples) * (self.X.T @ error)
        db = 2.0 * error.mean()

        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.W + self.b


exec(sys.stdin.read())