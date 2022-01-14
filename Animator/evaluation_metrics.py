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


def sum_of_squares(x, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    score = kmeans.score(x)
    return -score


def davis_bouldin(x, labels_pred):
    label_set = set(labels_pred)
    if len(label_set) == 1:
        return np.nan
    score = metrics.davies_bouldin_score(x, labels_pred)
    return score


def gmm_bayesian_information_criterion(x, labels_pred):
    label_set = set(labels_pred)
    gmm = GaussianMixture(n_components=len(label_set))
    gmm.fit(x)
    bic = gmm.bic(x)
    return bic


def lasso_bayesian_information_criterion_tsne(x, labels_pred):
    # dimensionality reduction
    dimensions = 3
    tsne = TSNE(n_components=dimensions, perplexity=40, n_iter=300)
    tsne_x = tsne.fit_transform(x)

    model_bic = LassoLarsIC(criterion='bic')
    model_bic.fit(tsne_x, labels_pred)
    alpha_bic_ = model_bic.alpha_
    return alpha_bic_


def gmm_bayesian_information_criterion_tsne(x, labels_pred):
    # dimensionality reduction
    dimensions = 3
    tsne = TSNE(n_components=dimensions, perplexity=40, n_iter=300)
    tsne_X = tsne.fit_transform(x)
    label_set = set(labels_pred)
    gmm = GaussianMixture(n_components=len(label_set))
    gmm.fit(tsne_X)
    bic = gmm.bic(tsne_X)
    return bic


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
