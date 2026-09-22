import matplotlib.pyplot as plt
import numpy as np
import util
from linear_model import LinearModel


class PoissonRegression(LinearModel):
    """Poisson regression with an exponential link."""
    def h(self, theta, x):
        return np.exp(x @ theta)

    def fit(self, x, y):
        """Fit by gradient ascent on the log likelihood."""
        def next_step(theta):
            return self.step_size / m * x.T @ (y-self.h(theta, x))

        m, n = x.shape

        if not self.theta:
            theta = np.zeros(n)
        else:
            theta = self.theta

        step = next_step(theta)
        while np.linalg.norm(step, 1) >= self.eps:
            theta += step
            step = next_step(theta)

        self.theta = theta

    def predict(self, x):
        return self.h(self.theta, x)


def plot(y_label, y_pred, title):
    plt.plot(y_label, 'go', label='label', markersize=4)
    plt.plot(y_pred, 'rx', label='prediction', markersize=3, linewidth=1)
    plt.suptitle(title, fontsize=12)
    plt.legend(loc='upper left')


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).resolve().parent)
    Path("output").mkdir(exist_ok=True)

    x_train, y_train = util.load_dataset('data/ds4_train.csv', add_intercept=True)

    x_valid, y_valid = util.load_dataset('data/ds4_valid.csv', add_intercept=True)

    poisson = PoissonRegression(step_size=2e-7)

    poisson.fit(x_train, y_train)

    y_train_pred = poisson.predict(x_train)

    plot(y_train, y_train_pred, 'Training Set')

    y_valid_pred = poisson.predict(x_valid)

    plot(y_valid, y_valid_pred, 'Validation Set')

    plt.plot(y_valid, y_valid_pred, 'bx', linewidth=2)

    plt.plot([0,5e7],[0,5e7], linewidth=1)

    plt.xlabel("true count")

    plt.ylabel("predict count")

    plt.show()
