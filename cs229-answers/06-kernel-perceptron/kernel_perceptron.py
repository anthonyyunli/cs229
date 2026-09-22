import math
import matplotlib.pyplot as plt
import numpy as np
import util


def plot(x, y, title):
    plt.figure()
    plt.plot(x[y==0, 0], x[y==0, 1], 'rx')
    plt.plot(x[y==1, 0], x[y==1, 1], 'bo')
    plt.suptitle(title, fontsize=12)


def sign(a):
    if a >= 0:
        return 1
    else:
        return 0


def dot_kernel(a, b):
    return np.dot(a, b)


def rbf_kernel(a, b, sigma=1):
    distance = (a - b).dot(a - b)
    scaled_distance = -distance / (2 * (sigma) ** 2)
    return math.exp(scaled_distance)


def initial_state():
    return []


def predict(state, kernel, x_i):
    """Evaluate the kernel expansion against the global training data."""
    z = 0
    for i in range(len(state)):
        z += state[i] * kernel(train_x[i], x_i)
    return 1 if z>= 0 else 0


def update_state(state, kernel, lr, x_i, y_i):
    """Append the next perceptron coefficient."""
    state.append(lr*(y_i - predict(state, kernel, x_i)))


def train_perceptron(kernel_name, kernel, learning_rate):
    """Train once through the data and save test predictions."""
    state = initial_state()

    for x_i, y_i in zip(train_x, train_y):
        update_state(state, kernel, learning_rate, x_i, y_i)

    plt.figure(figsize=(12, 8))
    util.plot_contour(lambda a: predict(state, kernel, a))
    util.plot_points(test_x, test_y)
    plt.plot()

    predict_y = [predict(state, kernel, test_x[i, :]) for i in range(test_y.shape[0])]

    np.savetxt('output/p05_{}_predictions'.format(kernel_name), predict_y)


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).resolve().parent)
    Path("output").mkdir(exist_ok=True)

    train_x, train_y = util.load_csv('data/ds5_train.csv')

    test_x, test_y = util.load_csv('data/ds5_test.csv')

    print(train_x, train_y)

    plot(train_x, train_y, "train")

    train_perceptron('dot', dot_kernel, 0.5)

    train_perceptron('rbf', rbf_kernel, 0.5)

    plt.show()
