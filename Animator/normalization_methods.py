import numpy as np
from sklearn.decomposition import PCA


def identity(data):
    """ no transformation """
    return data


def l2(data):
    return np.asarray(data)/(np.linalg.norm(data, axis=0) + 1e-4)


def l1(data):
    return np.asarray(data)/(np.linalg.norm(data, axis=0, ord=1) + 1e-4)


def min_max(data):
    raise NotImplementedError('this method was not implemented')


def pca_whitening_70d(data):
    # get_eigenvalues(data)
    dim = min(70, data.shape[0])
    model = PCA(whiten=True, n_components=dim)
    reduced_data = model.fit_transform(data)
    return reduced_data


def pca_whitening_50d(data):
    # get_eigenvalues(data)
    dim = min(50, data.shape[0])
    model = PCA(whiten=True, n_components=dim)
    reduced_data = model.fit_transform(data)
    return reduced_data


def pca_whitening_30d(data):
    # get_eigenvalues(data)
    dim = min(30, data.shape[0])
    model = PCA(whiten=True, n_components=dim)
    reduced_data = model.fit_transform(data)
    return reduced_data


def pca_whitening_10d(data):
    # get_eigenvalues(data)
    dim = min(10, data.shape[0])
    model = PCA(whiten=True, n_components=dim)
    reduced_data = model.fit_transform(data)
    return reduced_data


def get_eigenvalues(X):
    model = PCA(whiten=True).fit(X)
    n_samples = X.shape[0]

    # We center the data and compute the sample covariance matrix.
    mio = np.mean(X, axis=0)
    X -= mio
    cov_matrix = np.dot(X.T, X) / (n_samples - 1)
    eigen_vectors = model.components_
    return np.dot(eigen_vectors, np.dot(cov_matrix, eigen_vectors.T)).diagonal()
