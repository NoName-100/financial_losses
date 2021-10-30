import numpy as np
import pandas as pd

power = 4


def train(train_data):
    X = train_data ** (1 / power)
    mu = np.mean(X, axis=0)
    cov = np.cov(X.T)
    B = np.linalg.cholesky(cov)
    return mu, B


def generate_data(theta, Z):
    mu, B = theta
    generated_data = (mu + np.dot(Z, B.T)) ** power
    return generated_data


if __name__ == '__main__':
    # DATASET LOADING
    df = pd.read_csv('data/train.csv', header=None)
    train_data = np.array(df.drop(labels=0, axis=1))

    # TRAINING
    theta = train(train_data)

    # NOISE VECTOR LOADING
    try:
        Z = pd.read_csv('submission_data/noise_0.csv', header=None)
    except IOError:
        mu_Z = np.zeros(train_data.shape[1])
        cov_Z = np.eye(train_data.shape[1])
        Z = np.random.multivariate_normal(mu_Z, cov_Z, 410)

    # DATA GENERATION
    generated = generate_data(theta, Z)
    pd.DataFrame(generated).to_csv('submission_data/data.csv', index=False, header=False)
