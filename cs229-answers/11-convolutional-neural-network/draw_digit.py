from cnn import (MAX_POOL_SIZE, forward_convolution, forward_max_pool,
                 forward_relu, forward_linear, forward_softmax)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def predict(img_28):
    """Predict from a 28x28 image with pixel values in [0, 255]."""
    # Use the training normalization.
    x = (img_28 - TRAIN_MEAN) / TRAIN_STD

    x = x.reshape(1, 28, 28)

    conv = forward_convolution(
        params["W1"],
        params["b1"],
        x
    )

    pool = forward_max_pool(
        conv,
        MAX_POOL_SIZE,
        MAX_POOL_SIZE
    )

    relu = forward_relu(pool)

    flattened = relu.reshape(-1)

    logits = forward_linear(
        params["W2"],
        params["b2"],
        flattened
    )

    probs = forward_softmax(logits)

    return probs


def stamp(x, y):
    x = int(x)
    y = int(y)

    x0 = max(
        0,
        x - BRUSH_RADIUS
    )

    x1 = min(
        CANVAS_SIZE,
        x + BRUSH_RADIUS + 1
    )

    y0 = max(
        0,
        y - BRUSH_RADIUS
    )

    y1 = min(
        CANVAS_SIZE,
        y + BRUSH_RADIUS + 1
    )

    yy, xx = np.ogrid[
        y0:y1,
        x0:x1
    ]

    mask = (
        (xx - x) ** 2 +
        (yy - y) ** 2
        <= BRUSH_RADIUS ** 2
    )

    patch = canvas[
        y0:y1,
        x0:x1
    ]

    patch[mask] = 255


def draw_line(x0, y0, x1, y1):
    """Interpolate brush stamps to avoid gaps between mouse events."""
    distance = np.hypot(
        x1 - x0,
        y1 - y0
    )

    steps = max(
        1,
        int(
            distance /
            (BRUSH_RADIUS / 2)
        )
    )

    for t in np.linspace(
        0,
        1,
        steps + 1
    ):
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)

        stamp(x, y)


def on_press(event):
    global drawing
    global last_point

    if event.inaxes != ax:
        return

    if (
        event.xdata is None
        or event.ydata is None
    ):
        return

    drawing = True

    last_point = (
        event.xdata,
        event.ydata
    )

    stamp(
        event.xdata,
        event.ydata
    )

    image_display.set_data(canvas)

    fig.canvas.draw_idle()


def on_motion(event):
    global last_point

    if not drawing:
        return

    if event.inaxes != ax:
        return

    if (
        event.xdata is None
        or event.ydata is None
    ):
        return

    x = event.xdata
    y = event.ydata

    if last_point is not None:
        draw_line(
            last_point[0],
            last_point[1],
            x,
            y
        )

    last_point = (
        x,
        y
    )

    image_display.set_data(canvas)

    fig.canvas.draw_idle()


def on_release(event):
    global drawing
    global last_point

    drawing = False
    last_point = None


def clear(event):
    canvas[:] = 0

    image_display.set_data(
        canvas
    )

    ax.set_title(
        "Draw a digit"
    )

    network_img.set_data(
        np.zeros((28, 28))
    )

    network_ax.set_title(
        "What the network sees"
    )

    for bar, text in zip(
        prob_bars,
        prob_texts
    ):
        bar.set_height(0)

        text.set_y(
            0.02
        )

        text.set_text(
            "0.0%"
        )

    prob_ax.set_title(
        "Probabilities"
    )

    fig.canvas.draw_idle()
    results_fig.canvas.draw_idle()


def run_prediction(event):
    # Average each 10x10 canvas block into one input pixel.

    img_28 = canvas.reshape(
        28,
        10,
        28,
        10
    ).mean(
        axis=(1, 3)
    ).T

    probs = np.asarray(
        predict(img_28)
    ).ravel()

    predicted = int(
        np.argmax(probs)
    )

    confidence = (
        probs[predicted] * 100
    )

    ax.set_title(
        f"Prediction: {predicted} "
        f"({confidence:.1f}%)"
    )

    network_img.set_data(
        img_28.T
    )

    network_ax.set_title(
        "What the network sees\n"
        f"Prediction: {predicted}"
    )

    for digit, (
        bar,
        probability,
        text
    ) in enumerate(
        zip(
            prob_bars,
            probs,
            prob_texts
        )
    ):
        bar.set_height(
            probability
        )

        text.set_y(
            probability + 0.015
        )

        text.set_text(
            f"{probability * 100:.1f}%"
        )

    prob_ax.set_title(
        f"Prediction: {predicted} "
        f"({confidence:.1f}%)"
    )

    fig.canvas.draw_idle()

    results_fig.canvas.draw_idle()


if __name__ == "__main__":
    import os
    from pathlib import Path

    os.chdir(Path(__file__).resolve().parent)
    Path("output").mkdir(exist_ok=True)

    saved = np.load("output/model_weights.npz")

    params = {
        "W1": saved["W1"],
        "b1": saved["b1"],
        "W2": saved["W2"],
        "b2": saved["b2"],
    }

    if "mean" in saved.files and "std" in saved.files:
        TRAIN_MEAN = float(saved["mean"])
        TRAIN_STD = float(saved["std"])

    elif "mean" in globals() and "std" in globals():
        TRAIN_MEAN = float(mean)
        TRAIN_STD = float(std)

    else:
        raise RuntimeError(
            "Need the mean/std used during training. "
            "Save them with the model or define `mean` and `std` first."
        )

    CANVAS_SIZE = 280

    BRUSH_RADIUS = 12

    canvas = np.zeros(
        (CANVAS_SIZE, CANVAS_SIZE),
        dtype=np.float32
    )

    drawing = False

    last_point = None

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    fig.subplots_adjust(bottom=0.20)

    image_display = ax.imshow(
        canvas,
        cmap="gray",
        vmin=0,
        vmax=255,
        origin="upper"
    )

    ax.set_title("Draw a digit")

    ax.set_xticks([])

    ax.set_yticks([])

    results_fig, (
        network_ax,
        prob_ax
    ) = plt.subplots(
        1,
        2,
        figsize=(10, 4)
    )

    results_fig.suptitle(
        "Network Output"
    )

    network_img = network_ax.imshow(
        np.zeros((28, 28)),
        cmap="gray",
        vmin=0,
        vmax=255,
        origin="upper"
    )

    network_ax.set_title(
        "What the network sees"
    )

    network_ax.set_xticks([])

    network_ax.set_yticks([])

    digits = np.arange(10)

    prob_bars = prob_ax.bar(
        digits,
        np.zeros(10)
    )

    prob_ax.set_title(
        "Probabilities"
    )

    prob_ax.set_xlabel(
        "Digit"
    )

    prob_ax.set_ylabel(
        "Probability"
    )

    prob_ax.set_xticks(
        digits
    )

    prob_ax.set_ylim(
        0,
        1.10
    )

    prob_texts = []

    for digit in digits:
        text = prob_ax.text(
            digit,
            0.02,
            "0.0%",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=9
        )

        prob_texts.append(text)

    results_fig.tight_layout()

    clear_ax = fig.add_axes(
        [
            0.20,
            0.05,
            0.25,
            0.08
        ]
    )

    predict_ax = fig.add_axes(
        [
            0.55,
            0.05,
            0.25,
            0.08
        ]
    )

    clear_button = Button(
        clear_ax,
        "Clear"
    )

    predict_button = Button(
        predict_ax,
        "Predict"
    )

    clear_button.on_clicked(
        clear
    )

    predict_button.on_clicked(
        run_prediction
    )

    fig.canvas.mpl_connect(
        "button_press_event",
        on_press
    )

    fig.canvas.mpl_connect(
        "motion_notify_event",
        on_motion
    )

    fig.canvas.mpl_connect(
        "button_release_event",
        on_release
    )

    plt.show()
