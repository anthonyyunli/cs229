import matplotlib.pyplot as plt
import numpy as np
import util
from linear_model import LinearModel


class LocallyWeightedLinearRegression(LinearModel):
    """Local least squares with Gaussian distance weights."""
    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Store the training data for local fits at prediction time."""
        self.x = x
        self.y = y

    def predict(self, x):
        """Solve a weighted least-squares problem at each query point."""
        m, n = x.shape
        preds = np.zeros(m)

        for i in range(m):
            query = x[i] # (n, )
            diff = self.x - query # (m_train, n)
            w = np.exp(-np.sum(diff**2, axis=1) / (2 * self.tau**2)) # (m_train, )

            b = self.x.T @ (w[:, None] * self.x) # doing * to do fast diagonalization
            a = np.linalg.inv(b)
            theta = a @ self.x.T @ (w * self.y) # (n, )

            preds[i] = np.dot(theta, query)

        return preds


def plot(x, y_label, y_pred, title):
    plt.figure()
    plt.plot(x[:, -1], y_label, 'bx', linewidth=2, label='label')
    plt.plot(x[:, -1], y_pred, 'ro', linewidth=2, label='prediction')
    plt.suptitle(title, fontsize=12)
    plt.legend(loc='upper left')


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).resolve().parent)
    Path("output").mkdir(exist_ok=True)

    x_train, y_train = util.load_dataset('data/ds5_train.csv', add_intercept=True)

    x_valid, y_valid = util.load_dataset('data/ds5_valid.csv', add_intercept=True)

    x_test, y_test = util.load_dataset('data/ds5_test.csv', add_intercept=True)

    print(x_train.shape, x_train)

    plt.plot(x_train[:, -1], y_train, 'bx', linewidth=2)

    lwr = LocallyWeightedLinearRegression(0.05)

    lwr.fit(x_train, y_train)

    y_train_pred = lwr.predict(x_train)

    plot(x_train, y_train, y_train_pred, "Training Set")

    y_valid_pred = lwr.predict(x_valid)

    plot(x_valid, y_valid, y_valid_pred, "Validation Set")

    tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1]

    mse_best = np.inf

    tau_best = 0

    for tau in tau_values:
        lwr = LocallyWeightedLinearRegression(tau)
        lwr.fit(x_train, y_train)
        y_valid_pred = lwr.predict(x_valid)

        mse = np.mean((y_valid_pred - y_valid)**2)
        print(tau, mse)
        if mse < mse_best:
            tau_best = tau
            mse_best = mse

    print(f"best tau: {tau_best}, best mse: {mse_best}")

    lwr = LocallyWeightedLinearRegression(0.05)

    lwr.fit(x_train, y_train)

    y_test_pred = lwr.predict(x_test)

    plot(x_test, y_test, y_test_pred, "Training Set")

    print(np.mean((y_test_pred - y_test)**2))

    plt.show()
