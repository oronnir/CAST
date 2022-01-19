import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LassoLarsIC
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture


def silhouette_score(x, labels_pred):
    label_set_size = len(set(labels_pred))
    if label_set_size == 1:
        return 0.0

    if label_set_size == x.shape[0]:
        return 0.0

    silhouette = metrics.silhouette_score(x, labels_pred, metric='cosine')
    return silhouette


def purity(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    most_likely_assignment = np.amax(contingency_matrix, axis=0)

    # return purity
    return 1.0 * np.sum(most_likely_assignment) / np.sum(contingency_matrix)


def clustering_accuracy(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    most_likely_assignment = np.amax(contingency_matrix, axis=0)

    acc = np.average(1.0*most_likely_assignment / (1e-5 + np.sum(contingency_matrix, axis=0)))
    return acc


def itzi_k(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    n = y_true.shape[0]
    n_j = np.sum(contingency_matrix, axis=1)
    n_i = np.sum(contingency_matrix, axis=0)
    my_squarer = np.vectorize(lambda x: x ** 2)
    squared_contingency = my_squarer(contingency_matrix)
    acp = 1.0/n * np.sum(squared_contingency / n_i)
    aep = 1.0/n * np.sum(squared_contingency.T / n_j)
    return np.sqrt(acp*aep)


def f_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    n = y_true.shape[0]
    n_j = np.sum(contingency_matrix, axis=1)
    n_i = np.sum(contingency_matrix, axis=0)
    my_squarer = np.vectorize(lambda x: x ** 2)
    squared_contingency = my_squarer(contingency_matrix)
    acp = 1.0/n * np.sum(squared_contingency / n_i)
    aep = 1.0/n * np.sum(squared_contingency.T / n_j)
    return 2*acp*aep/(acp+aep)
