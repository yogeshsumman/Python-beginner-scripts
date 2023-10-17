import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# Example usage
if __name__ == "__main__":
    # Example data
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 3 + 4 * X + np.random.randn(100, 1)

    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    X_test = np.array([[2], [3]])
    predictions = model.predict(X_test)

    print(f"Predictions: {predictions}")
