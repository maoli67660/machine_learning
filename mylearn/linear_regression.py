import numpy as np

class LinearRegression():
    def __init__(self, alpha=0.03, n_iter=1500):

        self.alpha = alpha
        self.n_iter = n_iter
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X,y):
        self.n_samples = len(y)
        self.n_features = np.size(X, 1)
        self.params = np.zeros((self.n_features + 1, 1))
        
        # extend X matrix with 1 to learn  intercept_
        self.X = np.hstack((np.ones((self.n_samples, 1)), (X - np.mean(X, 0)) / np.std(X, 0)))
        self.y = y[:, np.newaxis]

        # gradient descent progress
        for _ in range(self.n_iter):
            self.params = self.params - (self.alpha/self.n_samples) * \
            self.X.T @ (self.X @ self.params - self.y)

        self.intercept_ = self.params[0]
        self.coef_ = self.params[1:]
        
        return self

    def predict(self, X):
        n_samples = np.size(X, 0)
        y = np.hstack((np.ones((n_samples, 1)), (X-np.mean(X, 0)) / np.std(X, 0))) @ self.params
        return y[:,0]

    def get_params(self):
        return self.params