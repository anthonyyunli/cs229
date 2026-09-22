# CS229 Answers

My solutions to the programming assignments from Stanford CS229 (Autumn 2018), in Python and NumPy. I originally worked in notebooks; the implementations are collected here as scripts, one folder per algorithm.

| Code | Method / example |
| --- | --- |
| [Logistic Regression](01-logistic-regression/logistic_regression.py) | Newton's method |
| [Gaussian Discriminant Analysis](02-gaussian-discriminant-analysis/gda.py) | Shared covariance, separate class means |
| [Positive-Only Learning](03-positive-only-learning/positive_only.py) | Logistic regression with missing positive labels |
| [Poisson Regression](04-poisson-regression/poisson_regression.py) | Count prediction with an exponential link |
| [Locally Weighted Linear Regression](05-locally-weighted-regression/locally_weighted_regression.py) | Gaussian weights and bandwidth selection |
| [Kernel Perceptron](06-kernel-perceptron/kernel_perceptron.py) | Dot-product and RBF kernels |
| [Naive Bayes](07-naive-bayes/naive_bayes.py) | SMS spam classification |
| [Step-Activation Neural Network](08-neural-network/neural_network.py) | Small network with hand-selected weights |
| [Gaussian Mixture Models](09-gaussian-mixture-model/gmm.py) | Unsupervised and semi-supervised EM |
| [K-Means](10-k-means/k_means.py) | Image compression with 16 colours |
| [Convolutional Neural Network](11-convolutional-neural-network/cnn.py) | NumPy CNN and handwritten-digit demo |
| [Independent Component Analysis](12-independent-component-analysis/ica.py) | Audio source separation |
| [CartPole](13-reinforcement-learning/cartpole.py) | Model-based control and value iteration |

## Running the code

From this directory, with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python 01-logistic-regression/logistic_regression.py
```

Each folder includes its data and a short README with a run command. Scripts write generated files to their own `output/` directory.

To try the handwritten-digit demo with the saved CNN weights:

```bash
python 11-convolutional-neural-network/draw_digit.py
```

The drawing window needs a desktop session with a graphical Matplotlib backend.

The assignments and datasets are from CS229. Supplied helpers (data loading, plotting, the base model class, the SVM comparison, and the CartPole environment) are marked in their files.
