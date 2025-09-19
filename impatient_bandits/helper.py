# impatient_contextual/helper.py
import numpy as np

class StickinessHelper:
    def __init__(self, prior_mvec, prior_cmat, noise_cmat=None):
        self.prior_mvec = prior_mvec
        self.prior_cmat = prior_cmat
        self.noise_cmat = noise_cmat

    @classmethod
    def from_data(cls, data):
        cmats = np.array([np.cov(mat.T) for mat in data.values()])
        noise_cmat = np.mean(cmats, axis=0)
        mvecs = np.array([np.mean(mat, axis=0) for mat in data.values()])
        prior_mvec = np.mean(mvecs, axis=0)
        prior_cmat = np.cov(mvecs.T)
        return cls(prior_mvec, prior_cmat, noise_cmat)