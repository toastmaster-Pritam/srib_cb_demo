# impatient_contextual/contextual_model.py
import numpy as np
from numpy.linalg import inv

class BayesianLinearModel:
    """
    Bayesian linear regression with Gaussian prior N(0, alpha*I) and known noise variance.
    Natural parameters: Lambda (precision) and b.
    """

    def __init__(self, dim, alpha=1.0, sigma2=0.25):
        self.dim = dim
        self.alpha = alpha
        self.sigma2 = sigma2
        self.Lambda = np.eye(dim) / alpha
        self.b = np.zeros(dim)

    def sample_theta(self):
        cov = inv(self.Lambda)
        mean = cov @ self.b
        return np.random.multivariate_normal(mean, cov)

    def update(self, x, y):
        x = x.reshape(-1)
        self.Lambda += np.outer(x, x) / self.sigma2
        self.b += (x * y) / self.sigma2

    def posterior_mean(self):
        return inv(self.Lambda) @ self.b