import numpy as np
import pandas as pd
import pickle
import random

random.seed(41)
np.random.seed(41)

power = 1.51
z_dim = 4


def train(data_file):
    df = pd.read_csv(data_file, header=None)
    train_data = np.array(df.drop(labels=0, axis=1))

    X = train_data ** (1 / power)
    mu = np.mean(X, axis=0)
    cov = np.cov(X.T)
    B = np.linalg.cholesky(cov)
    return mu, B, power


def G(theta, Z):
    mu, B, power = theta
    generated_data = np.abs(mu + np.dot(Z, B.T)) ** power
    return generated_data


if __name__ == '__main__':

    # THETA LOADING
    try:
        with open('theta.pth', 'rb') as theta_file:
            theta = pickle.load(theta_file)
    except FileNotFoundError:
        # Theta training
        theta = train('data/train.csv')

        file = open('theta.pth', 'wb')
        pickle.dump(theta, file)
        file.close()

    # NOISE VECTOR LOADING
    try:
        Z = pd.read_csv('noise.csv', header=None)
    except FileNotFoundError:
        mu_Z = np.zeros(z_dim)
        cov_Z = np.eye(z_dim)
        Z = np.random.multivariate_normal(mu_Z, cov_Z, 410)
        pd.DataFrame(Z).to_csv('noise.csv', index=False, header=False)

    # DATA GENERATION
    generated = G(theta, Z)
    pd.DataFrame(generated).to_csv('generated_samples.csv', index=False, header=False)
