import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import seaborn as sns

sns.set_style('white')
sns.set()


class logistic_regresion:
    def __init__(self, iterations=1500, learning_rate=0.03):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.loss_history = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_cost(self, X, y, params):
        h = self.sigmoid(X @ params)
        epsilon = 1e-5
        cost = (1 / self.n_samples) * (((-y).T @ np.log(h + epsilon)) - ((1 - y).T @ np.log(1 - h + epsilon)))
        return cost

    def gradient_descent(self, X, y):
        for i in range(self.iterations):
            self.params = self.params - (self.learning_rate / self.n_samples) * (
                    X.T @ (self.sigmoid(X @ self.params) - y))
            self.loss_history.append(self.compute_cost(X, y, self.params)[0][0])

    def fit(self, X, y):
        # print(f'self.n_features = {self.n_features}  self.n_samples={self.n_samples} , X.shape:{X.shape}')
        self.n_samples = len(y)
        X = np.hstack((np.ones((self.n_samples, 1)), X))
        self.n_features = len(X[0])
        self.params = np.zeros((self.n_features, 1))
        assert len(X) == self.n_samples
        self.gradient_descent(X, y)
        return self

    def predict(self, X, params):
        return np.round(self.sigmoid(X @ params))


if __name__ == '__main__':
    X, y = make_classification(n_samples=500, n_features=2,
                               n_redundant=0, n_informative=1,
                               n_clusters_per_class=1, random_state=14)
    y = y[:, np.newaxis]

    cls = logistic_regresion()
    cls.fit(X, y)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu_r')

    opt_para = cls.params[:, 0]
    print(f' opt_para =  {opt_para}')
    # print('lost history: ', cls.loss_history)
    slope = -(opt_para[1] / opt_para[2])
    intercept = -(opt_para[0] / opt_para[2])
    line_x = np.linspace(1.7, 3.4, 100) / 20
    print(f'slope = {slope}     intercept={intercept}')
    plt.plot(line_x, intercept + (slope * line_x), c='g', lw=3)
    plt.show()
