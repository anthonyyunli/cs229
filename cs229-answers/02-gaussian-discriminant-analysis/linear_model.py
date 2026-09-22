# CS229-provided base model class.


class LinearModel(object):
    """Base class for linear models."""
    def __init__(self, step_size=0.2, max_iter=100, eps=1e-5,
                 theta_0=None, verbose=True):
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run solver to fit linear model."""
        raise NotImplementedError('Subclass of LinearModel must implement fit method.')

    def predict(self, x):
        raise NotImplementedError('Subclass of LinearModel must implement predict method.')
