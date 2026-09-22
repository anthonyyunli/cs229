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


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).resolve().parent)
    Path("output").mkdir(exist_ok=True)

    ds1_training_set_path = 'data/ds1_train.csv'

    ds1_valid_set_path = 'data/ds1_valid.csv'

    ds2_training_set_path = 'data/ds2_train.csv'

    ds2_valid_set_path = 'data/ds2_valid.csv'

    x_train, y_train = util.load_dataset(ds1_training_set_path, add_intercept=True)

    x_valid, y_valid = util.load_dataset(ds1_valid_set_path, add_intercept=True)

    plt.plot(x_train[y_train == 1, -2], x_train[y_train == 1, -1], 'bx', linewidth=2)

    plt.plot(x_train[y_train == 0, -2], x_train[y_train == 0, -1], 'go', linewidth=2)

    log_reg = LogisticRegression()

    for i in range(100):
        log_reg.fit(x_train, y_train)

    util.plot(x_train, y_train, log_reg.theta)

    print("Theta = ", log_reg.theta)

    print("Accuracy is", np.mean(log_reg.predict(x_train)==y_train))

    util.plot(x_valid, y_valid, log_reg.theta)

    print("Accuracy is", np.mean(log_reg.predict(x_valid)==y_valid))

    plt.show()
