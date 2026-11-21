import sys

import numpy as np


def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))


def log_loss(w, b, X, y):
    scores = X @ w + b
    probabilities = sigmoid(scores)

    # Средняя логарифмическая потеря для меток 0 и 1
    loss = np.logaddexp(0.0, (1.0 - 2.0 * y) * scores).mean()

    # Градиенты по весам и свободному коэффициенту
    errors = probabilities - y
    dw = X.T @ errors / y.size
    db = errors.mean()

    return loss, (dw, db)


def optimize(w, b, X, y, n_iterations, eta):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)
    losses = []

    for _ in range(n_iterations):
        loss, (dw, db) = log_loss(w, b, X, y)
        losses.append(loss)

        w = w - eta * dw
        b = b - eta * db

    return w, b, losses


def predict(w, b, X, p=0.5):
    probabilities = sigmoid(X @ w + b)
    return (probabilities > p).astype(int)


exec(sys.stdin.read())