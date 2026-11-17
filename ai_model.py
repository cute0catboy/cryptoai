import numpy as np
import pandas as pd

class LogisticRegression:

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = 0

    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))

    def predict_proba(self, X: pd.DataFrame):

        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def compute_loss(self, y, p):

        epsilon = 1e-15
        p = np.clip(p, epsilon, 1-epsilon)

        loss = -np.mean(
            y * np.log(p) + (1 - y) * np.log(1 - p)
        )

        return loss

    def fit(self, X: pd.DataFrame, y: pd.Series):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):

            #предсказание
            prediction = self.predict_proba(X)
            error = prediction - y

            #обновление d
            dw = np.dot(X.T, error) / n_samples
            db = np.sum(error) / n_samples

            #сам градиентный спуск
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            #счетчик эпох
            if epoch % 100 == 0:
                loss = self.compute_loss(y, prediction)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X, threshold: float = 0.5):

        probabilities = self.predict_proba(X)
        predictions = (probabilities >= threshold).astype(int)
        return predictions
