import matplotlib.pyplot as plt
import numpy as np
import util
from linear_model import LinearModel


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's method."""
    def fit(self, x, y):
        """Perform one Newton update; x includes an intercept column."""
        def h(theta, x):
            return 1 / (1+np.exp(-(x @ theta)))

        def gradient(theta, x, y):
            m, _ = x.shape
            return -1/m * x.T @ (y-h(theta, x))

        def hessian(theta, x):
            m, _ = x.shape
            return 1/m * x.T @ np.diag(h(theta, x) * (1-h(theta,x))) @ x

        m,n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)
        self.theta = self.theta - np.linalg.inv(hessian(self.theta, x)) @ gradient(self.theta, x, y)

    def predict(self, x):
        return x @ self.theta >= 0


def h(theta, x):
    return 1 / (1+np.exp(-(x @ theta)))


def predict(theta, x):
    return h(theta, x) / a >= 0.5


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).resolve().parent)
    Path("output").mkdir(exist_ok=True)

    x_train, y_train = util.load_dataset('data/ds3_train.csv', add_intercept=True)

    _, t_train = util.load_dataset('data/ds3_train.csv', label_col='t')

    x_valid, y_valid = util.load_dataset('data/ds3_valid.csv', add_intercept=True)

    _, t_valid = util.load_dataset('data/ds3_valid.csv', label_col='t')

    x_test, y_test = util.load_dataset('data/ds3_test.csv', add_intercept=True)

    _, t_test = util.load_dataset('data/ds3_test.csv', label_col='t')

    plt.figure()

    plt.plot(x_train[y_train == 0, -2], x_train[y_train == 0, -1], 'go', linewidth=2)

    plt.plot(x_train[y_train == 1, -2], x_train[y_train == 1, -1], 'bx', linewidth=2)

    log_reg = LogisticRegression()

    for i in range(100):
        log_reg.fit(x_train, t_train)

    util.plot(x_train, t_train, log_reg.theta)

    print("Theta = ", log_reg.theta)

    print("Accuracy is", np.mean(log_reg.predict(x_train)==t_train))

    util.plot(x_valid, t_valid, log_reg.theta)

    print("Theta = ", log_reg.theta)

    print("Accuracy is", np.mean(log_reg.predict(x_valid)==t_valid))

    log_reg = LogisticRegression()

    for i in range(100):
        log_reg.fit(x_train, y_train)

    util.plot(x_test, y_test, log_reg.theta)

    print("Theta = ", log_reg.theta)

    print("Accuracy is", np.mean(log_reg.predict(x_test)==y_test))

    util.plot(x_valid, y_valid, log_reg.theta)

    print("Theta = ", log_reg.theta)

    print("Accuracy is", np.mean(log_reg.predict(x_valid)==y_valid))

    a = np.sum(h(log_reg.theta, x_valid) * y_valid) / np.sum(y_valid)

    print(a)

    theta_prime = log_reg.theta + np.log(2/a-1) * np.array([1,0,0])

    util.plot(x_valid, t_valid, theta_prime)

    print("Theta = ", theta_prime)

    print("Accuracy is", np.mean(predict(theta_prime, x_valid)==y_valid))

    plt.show()
