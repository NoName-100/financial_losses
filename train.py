import numpy as np


def train(train_data):
    power = 4

    X = train_data ** (1 / power)
    mu = np.mean(X, axis=0)
    cov = np.cov(X.T)
    B = np.linalg.cholesky(cov)
    return mu, B
