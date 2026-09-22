import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path

MAX_POOL_SIZE = 5
CONVOLUTION_SIZE = 4
CONVOLUTION_FILTERS = 2


def forward_softmax(x):
    """Subtract the largest logit before exponentiating to avoid overflow."""
    x = x - np.max(x,axis=0)
    exp = np.exp(x)
    s = exp / np.sum(exp,axis=0)
    return s


def backward_softmax(x, grad_outputs):
    """Softmax gradient for the one-hot cross-entropy loss used here."""
    return forward_softmax(x) - (grad_outputs!=0).astype(int)


def forward_relu(x):
    x[x<=0] = 0

    return x


def backward_relu(x, grad_outputs):
    grad_outputs[x<=0] = 0
    return grad_outputs


def get_initial_params():
    """Initialize convolution and output weights, with zero biases."""
    size_after_convolution = 28 - CONVOLUTION_SIZE + 1
    size_after_max_pooling = size_after_convolution // MAX_POOL_SIZE

    num_hidden = size_after_max_pooling * size_after_max_pooling * CONVOLUTION_FILTERS

    return {
        'W1': np.random.normal(size = (CONVOLUTION_FILTERS, 1, CONVOLUTION_SIZE, CONVOLUTION_SIZE), scale=1/ math.sqrt(CONVOLUTION_SIZE * CONVOLUTION_SIZE)),
        'b1': np.zeros(CONVOLUTION_FILTERS),
        'W2': np.random.normal(size = (num_hidden, 10), scale = 1/ math.sqrt(num_hidden)),
        'b2': np.zeros(10)
    }


def forward_convolution(conv_W, conv_b, data):
    """Valid convolution: data is (channels, width, height)."""
    conv_channels, _, conv_width, conv_height = conv_W.shape

    input_channels, input_width, input_height = data.shape

    output = np.zeros((conv_channels, input_width - conv_width + 1, input_height - conv_height + 1))

    for x in range(input_width - conv_width + 1):
        for y in range(input_height - conv_height + 1):
            for output_channel in range(conv_channels):
                output[output_channel, x, y] = np.sum(
                    np.multiply(data[:, x:(x + conv_width), y:(y + conv_height)], conv_W[output_channel, :, :, :])) + conv_b[output_channel]

    return output


def backward_convolution(conv_W, conv_b, data, output_grad):
    """Return gradients for convolution weights, bias, and input."""
    c_ch, _, c_w, c_h= conv_W.shape
    in_ch, in_w, in_h = data.shape

    grad_b = output_grad.sum(axis=(1,2))

    grad_W = np.zeros_like(conv_W) # (c_ch, in_ch, c_w, c_h)
    grad_data = np.zeros_like(data) # (in_ch, in_w, in_h)

    for x in range(in_w - c_w + 1):
        for y in range(in_h - c_h + 1):
            for c in range(c_ch):
                grad_W[c, :, :, :] += data[:, x:(x+c_w), y:(y+c_h)] * output_grad[c, x, y]
                grad_data[:, x:(x+c_w), y:(y+c_h)] += conv_W[c, :, :, :] * output_grad[c, x, y]

    return grad_W, grad_b, grad_data


def forward_max_pool(data, pool_width, pool_height):
    """Max pooling with stride equal to the pool size."""
    input_channels, input_width, input_height = data.shape

    output = np.zeros((input_channels, input_width // pool_width, input_height // pool_height))

    for x in range(0, input_width, pool_width):
        for y in range(0, input_height, pool_height):
            output[:, x // pool_width, y // pool_height] = np.amax(data[:, x:(x + pool_width), y:(y + pool_height)], axis=(1, 2))

    return output


def backward_max_pool(data, p_w, p_h, output_grad):
    """Route each gradient to the maximum in its pooling window."""
    c, w, h = data.shape

    grad_data = np.zeros((c, w, h))

    for x in range(0, w, p_w):
        for y in range(0, h, p_h):
                patch = data[:, x:(x+p_w), y:(y+p_h)].reshape(c, -1)
                l_x, l_y = np.unravel_index(np.argmax(patch, axis=-1), (p_w, p_h))
                grad_data[np.arange(c), x+l_x, y+l_y] += output_grad[:, x//p_w, y//p_h]

    return grad_data


def forward_cross_entropy_loss(probabilities, labels):
    result = 0

    for i, label in enumerate(labels):
        if label == 1:
            result += -np.log(probabilities[i])

    return result


def backward_cross_entropy_loss(probabilities, labels):
    return -labels / probabilities


def forward_linear(weights, bias, data):
    return data.dot(weights) + bias


def backward_linear(weights, bias, data, output_grad):
    """Return gradients for weights, bias, and input."""
    grad_weights = np.outer(data, output_grad)
    grad_bias = output_grad
    grad_data = weights @ output_grad

    return grad_weights, grad_bias, grad_data


def forward_prop(data, labels, params):
    """Return class probabilities and loss for one (1, 28, 28) image."""
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    first_convolution = forward_convolution(W1, b1, data)
    first_max_pool = forward_max_pool(first_convolution, MAX_POOL_SIZE, MAX_POOL_SIZE)
    first_after_relu = forward_relu(first_max_pool)

    flattened = np.reshape(first_after_relu, (-1))

    logits = forward_linear(W2, b2, flattened)

    y = forward_softmax(logits)
    cost = forward_cross_entropy_loss(y, labels)

    return y, cost


def backward_prop(data, labels, params):
    """Return parameter gradients, class probabilities, and loss."""
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    first_convolution = forward_convolution(W1, b1, data)
    first_max_pool = forward_max_pool(first_convolution, MAX_POOL_SIZE, MAX_POOL_SIZE)

    first_after_relu = forward_relu(first_max_pool)
    flattened = np.reshape(first_after_relu, (-1))

    logits = forward_linear(W2, b2, flattened)
    y = forward_softmax(logits)

    cost = forward_cross_entropy_loss(y, labels)

    grad_CE = backward_cross_entropy_loss(y, labels)
    grad_softmax = backward_softmax(logits, grad_CE)
    grad_W2, grad_b2, grad_linear = backward_linear(W2, b2, flattened, grad_softmax)
    grad_linear = grad_linear.reshape(first_after_relu.shape)
    grad_relu = backward_relu(first_max_pool, grad_linear)
    grad_max_pool = backward_max_pool(first_convolution, MAX_POOL_SIZE, MAX_POOL_SIZE, grad_relu)
    grad_W1, grad_b1, grad_data = backward_convolution(W1, b1, data, grad_max_pool)

    return {
        "W1": grad_W1,
        "b1": grad_b1,
        "W2": grad_W2,
        "b2": grad_b2
    }, y, cost


def forward_prop_batch(batch_data, batch_labels, params, forward_prop_func):
    """Apply the forward pass to each image in a batch."""
    y_array = []
    cost_array = []

    for item, label in zip(batch_data, batch_labels):
        y, cost = forward_prop_func(item, label, params)
        y_array.append(y)
        cost_array.append(cost)

    return np.array(y_array), np.array(cost_array)


def gradient_descent_batch(batch_data, batch_labels, learning_rate, params, backward_prop_func):
    """Update parameters using batch gradients; return mean loss and accuracy."""
    total_grad = {}

    total_loss = 0
    correct = 0

    for i in range(batch_data.shape[0]):
        grad, y, cost = backward_prop_func(
            batch_data[i, :, :],
            batch_labels[i, :],
            params)

        for key, value in grad.items():
            if key not in total_grad:
                total_grad[key] = np.zeros(value.shape)

            total_grad[key] += value

        total_loss += cost

        if np.argmax(y) == np.argmax(batch_labels[i]):
            correct += 1

    params['W1'] = params['W1'] - learning_rate * total_grad['W1']
    params['W2'] = params['W2'] - learning_rate * total_grad['W2']
    params['b1'] = params['b1'] - learning_rate * total_grad['b1']
    params['b2'] = params['b2'] - learning_rate * total_grad['b2']

    n = batch_data.shape[0]

    return total_loss / n, correct / n


def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels


def read_data(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')

    x = np.reshape(x, (x.shape[0], 1, 28, 28))

    return x, y


def load_training_checkpoint(weights_path):
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    with np.load(weights_path) as saved:
        required = ("W1", "b1", "W2", "b2", "mean", "std")

        missing = [key for key in required if key not in saved.files]
        if missing:
            raise ValueError(f"Missing checkpoint entries: {missing}")

        params = {
            "W1": saved["W1"].copy(),
            "b1": saved["b1"].copy(),
            "W2": saved["W2"].copy(),
            "b2": saved["b2"].copy(),
        }

        mean = float(saved["mean"])
        std = float(saved["std"])

    # Check that the saved weights match this architecture.
    size_after_convolution = 28 - CONVOLUTION_SIZE + 1
    size_after_max_pooling = size_after_convolution // MAX_POOL_SIZE

    num_hidden = (
        size_after_max_pooling
        * size_after_max_pooling
        * CONVOLUTION_FILTERS
    )

    expected_shapes = {
        "W1": (
            CONVOLUTION_FILTERS,
            1,
            CONVOLUTION_SIZE,
            CONVOLUTION_SIZE,
        ),
        "b1": (CONVOLUTION_FILTERS,),
        "W2": (num_hidden, 10),
        "b2": (10,),
    }

    for key, expected_shape in expected_shapes.items():
        if params[key].shape != expected_shape:
            raise ValueError(
                f"{key} has shape {params[key].shape}, "
                f"expected {expected_shape}"
            )

    if std == 0:
        raise ValueError("Saved training std is 0.")

    return params, mean, std


def nn_train(
    train_data,
    train_labels,
    dev_data,
    dev_labels,
    get_initial_params_func,
    forward_prop_func,
    backward_prop_func,
    learning_rate=1e-2,
    batch_size=16,
    num_epochs=1,
    initial_params=None,
    print_every=100
):
    if initial_params is None:
        params = get_initial_params_func()
    else:
        params = {
            key: value.copy()
            for key, value in initial_params.items()
        }

    m = train_data.shape[0]

    # Include the final partial batch.
    num_batches = (m + batch_size - 1) // batch_size

    for epoch in range(num_epochs):
        for batch in range(num_batches):
            start = batch * batch_size
            end = min(start + batch_size, m)

            batch_data = train_data[start:end]
            batch_labels = train_labels[start:end]

            loss, accuracy = gradient_descent_batch(
                batch_data,
                batch_labels,
                learning_rate,
                params,
                backward_prop_func
            )

            if (
                batch == 0
                or (batch + 1) % print_every == 0
                or batch == num_batches - 1
            ):
                print(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"batch {batch + 1}/{num_batches} | "
                    f"loss={loss:.4f} | "
                    f"accuracy={accuracy:.3f}"
                )

    return params


def nn_test(data, labels, params):
    output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy


def compute_accuracy(output, labels):
    correct_output = np.argmax(output,axis=1)
    correct_labels = np.argmax(labels,axis=1)

    is_correct = [a == b for a,b in zip(correct_output, correct_labels)]

    accuracy = sum(is_correct) * 1. / labels.shape[0]
    return accuracy


def run_train(
    all_data,
    all_labels,
    backward_prop_func,
    initial_params=None,
    mean=None,
    std=None
):
    params = nn_train(
        all_data["train"],
        all_labels["train"],
        all_data["dev"],
        all_labels["dev"],
        get_initial_params,
        forward_prop,
        backward_prop_func,

        learning_rate=1e-2,
        batch_size=16,
        num_epochs=1,

        initial_params=initial_params,

        print_every=100
    )

    Path("output").mkdir(exist_ok=True)

    np.savez(
        "output/model_weights.npz",
        W1=params["W1"],
        b1=params["b1"],
        W2=params["W2"],
        b2=params["b2"],
        mean=mean,
        std=std,
    )

    print("Saved model to output/model_weights.npz")

    return params


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).resolve().parent)
    Path("output").mkdir(exist_ok=True)

    LOAD_WEIGHTS = True

    WEIGHTS_PATH = "output/model_weights.npz"

    np.random.seed(100)

    train_data, train_labels = read_data(
        'data/images_train.csv.gz',
        'data/labels_train.csv.gz'
    )

    train_labels = one_hot_labels(train_labels)

    p = np.random.permutation(60000)

    train_data = train_data[p, :]

    train_labels = train_labels[p, :]

    dev_data = train_data[0:400, :]

    dev_labels = train_labels[0:400, :]

    train_data = train_data[400:, :]

    train_labels = train_labels[400:, :]

    if LOAD_WEIGHTS:
        initial_params, mean, std = load_training_checkpoint(
            WEIGHTS_PATH
        )

        print(f"Loaded weights from {WEIGHTS_PATH}")

    else:
        initial_params = None

        mean = np.mean(train_data)
        std = np.std(train_data)

        print("Starting with new weights")

    train_data = (train_data - mean) / std

    dev_data = (dev_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
    }

    plt.show()
