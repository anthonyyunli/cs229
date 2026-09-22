# Convolutional Neural Network

A NumPy CNN for handwritten digits: convolution, max pooling, ReLU, and a softmax output. `cnn.py` contains the forward pass, backpropagation, and training functions. Running it loads the saved checkpoint and prepares MNIST; it does not start training.

The data is included as compressed CSV files, which NumPy reads directly. The trained weights are in `output/model_weights.npz`.

Run:

```bash
python 11-convolutional-neural-network/cnn.py
```

For the drawing demo, run this in a desktop session:

```bash
python 11-convolutional-neural-network/draw_digit.py
```
