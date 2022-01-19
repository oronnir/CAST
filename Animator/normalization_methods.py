import numpy as np
from sklearn.decomposition import PCA


def identity(data):
    """ no transformation """
    return data


def l2(data):
    return np.asarray(data)/(np.linalg.norm(data, axis=0) + 1e-4)


def l1(data):
    return np.asarray(data)/(np.linalg.norm(data, axis=0, ord=1) + 1e-4)


def pca_whitening_30d(data):
    # get_eigenvalues(data)
    dim = min(30, data.shape[0])
    model = PCA(whiten=True, n_components=dim)
    reduced_data = model.fit_transform(data)
    return reduced_data
