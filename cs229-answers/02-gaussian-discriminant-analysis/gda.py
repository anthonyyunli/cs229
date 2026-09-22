import matplotlib.pyplot as plt
import numpy as np
import util
from linear_model import LinearModel


class GDA(LinearModel):
    """Gaussian discriminant analysis with a shared covariance matrix."""
    def fit(self, x, y):
        """Estimate class means and shared covariance; x includes an intercept."""
        x_raw = x[:, 1:]

        m,n = x_raw.shape
        phi = 1/m * np.sum(y==1)

        mu_0 = np.sum(x_raw[y==0], 0)/np.sum(y==0)
        mu_1 = np.sum(x_raw[y==1], 0)/np.sum(y==1)

        centered = np.zeros_like(x_raw)
        centered[y==0] = x_raw[y==0] - mu_0
        centered[y==1] = x_raw[y==1] - mu_1

        sigma = 1/m * centered.T @ centered
        print(sigma, sigma.shape)

        theta = (mu_1-mu_0) @ np.linalg.inv(sigma)
        theta_0 = 1/2 * (mu_0+mu_1).T @ np.linalg.inv(sigma) @ (mu_0-mu_1) - np.log((1-phi)/phi)

        self.theta = np.insert(theta, 0 , theta_0)

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

    gda = GDA()

    gda.fit(x_train, y_train)

    util.plot(x_train, y_train, gda.theta)

    print("Theta = ", gda.theta)

    print("Accuracy is", np.mean(gda.predict(x_train)==y_train))

    util.plot(x_valid, y_valid, gda.theta)

    print("Accuracy is", np.mean(gda.predict(x_valid)==y_valid))

    plt.show()
