import math
import numpy as np
from itertools import groupby
from collections import Counter
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from Animator.modularity_maximizer import EntityGraph
from numpy import linalg as LA


def c_optics(X, k, frame_ids, tube_ids=None):
    distances = fast_cosine_distance(X)
    same_frame = fast_mask(frame_ids)
    same_tube = fast_mask(tube_ids) if tube_ids is not None else 0
    constrained_pairwise_distances = distances + same_frame - same_tube
    clustering_obj = OPTICS(min_samples=7, min_cluster_size=7, metric="precomputed", algorithm='brute')\
        .fit(constrained_pairwise_distances)

    # find centers
    cluster_centers, best_ids = get_cluster_centers(X, clustering_obj.labels_)
    return clustering_obj.labels_, cluster_centers


def estimate_eps_with_knn(precomputed_distances: np.ndarray) -> float:
    """
    Note: The following method under performs on the animation data...

    estimate the best epsilon parameter for DBSCAN using the K Nearest Neighbor Algorithm.
    Based on the following paper:
    @inproceedings{rahmah2016determination,
  title={Determination of optimal epsilon (eps) value on dbscan algorithm to clustering data on peatland hotspots in
  sumatra},
  author={Rahmah, Nadia and Sitanggang, Imas Sukaesih},
  booktitle={IOP conference series: earth and environmental science},
  volume={31},
  number={1},
  pages={012012},
  year={2016},
  organization={IOP Publishing}
}

    :param precomputed_distances: The precomputed distances
    :return: The optimal epsilon for DBSCAN
    """
    from sklearn.neighbors import NearestNeighbors
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set()

    knn = NearestNeighbors(n_neighbors=2, metric='precomputed')
    nearest_neighbors_fitted = knn.fit(precomputed_distances)
    distances, indices = nearest_neighbors_fitted.kneighbors(precomputed_distances)

    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    elbow_ = plt.plot(distances)
    elbow_[0].figure.show()

    from kneed import KneeLocator
    y = distances
    x = range(1, len(distances)+1)
    kn = KneeLocator(x, y, curve='convex', direction='increasing')
    print(f'Best knee is {kn.knee}')

    plt.xlabel('number of clusters k')
    plt.ylabel('Sum of squared distances')
    plt.plot(x, y, 'bx-')
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    theoretical_eps = kn.knee
    return theoretical_eps


def precompute_distances(x, frame_ids, tube_ids):
    distances = pairwise_distances(x, metric='cosine')
    same_frame = fast_mask(frame_ids)
    same_tube = fast_mask(tube_ids)
    constrained_pairwise_distances = np.maximum(distances, same_frame - same_tube)
    return constrained_pairwise_distances


def c_dbscan(x, k, frame_ids=None, tube_ids=None, eps_span=13.0, initial_eps=5.001):
    counter = 1
    max_iterations = 12
    drill_down_iterations = 12
    best_clusters = None
    best_k = 0
    k_tolerance = 0.2*k
    is_best = False
    best_eps = initial_eps
    from_eps = best_eps
    epsilon = None

    # distance
    constrained_pairwise_distances = precompute_distances(x, frame_ids, tube_ids)

    for j in range(drill_down_iterations):
        eps_step = eps_span / max_iterations
        prev_k = -1
        before_prev_k = -1
        for i in range(max_iterations):
            epsilon = from_eps + 1.*i*eps_step
            clustering = DBSCAN(eps=epsilon, min_samples=4, p=2.0, n_jobs=-1, metric='precomputed')\
                .fit(constrained_pairwise_distances)
            clusters = clustering.labels_

            actual_k = len(set([c for c in clusters if c >= 0]))
            if abs(actual_k-k) < abs(best_k-k):
                best_k = actual_k
                best_clusters = clusters
                best_eps = epsilon
            if actual_k - k_tolerance <= k <= actual_k + k_tolerance:
                print('DBSCAN yielded {} close enough to the estimation with eps={}, k={} - stopping the search...'
                      .format(actual_k, epsilon, k))
                is_best = True
                break
            print('DBSCAN yielded {} clusters while the estimation was {}. Epsilon={}'.format(actual_k, k, epsilon))
            counter += 1

            # when the epsilon is converged into 1 no point to continue
            if (prev_k >= 1 and actual_k == 1) or (before_prev_k > prev_k > actual_k):
                print('overshoot...')
                break
            before_prev_k = prev_k
            prev_k = actual_k

        if is_best:
            break
        eps_span = 2.0 * eps_step

        from_eps = epsilon if best_k == 0 else max(1e-2, best_eps - .95 * eps_step)

    best_k = len(set([c for c in best_clusters if c >= 0]))
    print('actual k by DBSCAN is {} with epsilon={}'.format(best_k, best_eps))

    # find centers
    cluster_centers, best_ids = get_cluster_centers(x, best_clusters)
    return best_clusters, cluster_centers


def fast_cosine_distance(X):
    # base similarity matrix (all dot products)
    similarity = np.dot(X, X.T)

    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine_sim = cosine.T * inv_mag

    # similarity to distance
    distance = 1. - cosine_sim

    # clipping
    distance[distance < 0] = 0.
    distance[distance > 1] = 1.
    return distance


def fast_mask(ids):
    a = np.stack([ids for j in range(len(ids))], axis=0)
    return np.asarray(a == a.T, dtype=float)


def k_means(x, k, frame_ids=None):
    """ vanilla KMeans """
    kmeans = KMeans(n_clusters=k, n_init=100, max_iter=2000, random_state=1234567)
    kmeans.fit(x)
    cluster_ids = kmeans.predict(x)
    return cluster_ids, kmeans.cluster_centers_


def affinity_propagation(x, k, frame_ids=None):
    distances = fast_cosine_distance(x)
    same_frame = fast_mask(frame_ids)
    constrained_pairwise_distances = distances + same_frame
    ap = AffinityPropagation(affinity='precomputed', damping=.5, max_iter=500, convergence_iter=50)
    ap = ap.fit(constrained_pairwise_distances)
    cluster_ids = ap.labels_

    # map singletons to -1
    lab_hist = Counter(cluster_ids)
    cluster_ids = [lab if lab_hist[lab] > 1 else -1 for lab in cluster_ids]

    # get cluster centers
    cluster_centers, best_ids = get_cluster_centers(x, cluster_ids)

    return cluster_ids, cluster_centers


def k_means_k30(x, k, frame_ids=None):
    """ vanilla KMeans """
    kmeans = KMeans(n_clusters=k)
    print(x)
    kmeans.fit(x)
    cluster_ids = kmeans.predict(x)
    return cluster_ids, kmeans.cluster_centers_


def dbscan(x, k, frame_ids=None, eps_span=13.0, initial_eps=12.001):
    counter = 1
    max_iterations = 7
    drill_down_iterations = 4
    best_clusters = None
    best_k = 0
    k_tolerance = 0.2*k
    is_best = False
    best_eps = initial_eps
    from_eps = best_eps
    epsilon = None
    eps_step = -1
    for j in range(drill_down_iterations):
        eps_step = eps_span / max_iterations
        prev_k = -1
        before_prev_k = -1
        for i in range(max_iterations):
            epsilon = from_eps + 1.*i*eps_step
            clustering = DBSCAN(eps=epsilon, min_samples=3, p=2.0, n_jobs=-1, metric='l2').fit(x)
            clusters = clustering.labels_

            actual_k = len(set([c for c in clusters if c >= 0]))
            if abs(actual_k-k) < abs(best_k-k):
                best_k = actual_k
                best_clusters = clusters
                best_eps = epsilon
            if actual_k - k_tolerance <= k <= actual_k + k_tolerance:
                print('DBSCAN yielded {} close enough to the estimation with eps={}, k={} - stopping the search...'
                      .format(actual_k, epsilon, k))
                is_best = True
                break
            print('DBSCAN yielded {} clusters while the estimation was {}. Epsilon={}'.format(actual_k, k, epsilon))
            counter += 1

            # when the epsilon is converged into 1 no point to continue
            if (prev_k >= 1 and actual_k == 1) or (before_prev_k > prev_k > actual_k):
                print('overshoot...')
                break
            before_prev_k = prev_k
            prev_k = actual_k

        if is_best:
            break
        eps_span = 2.0 * eps_step

        from_eps = epsilon if best_k == 0 else max(1e-2, best_eps - .95 * eps_step)

    best_k = len(set([c for c in best_clusters if c >= 0]))
    print('actual k by DBSCAN is {} with epsilon={}'.format(best_k, best_eps))

    # find centers
    cluster_centers, best_ids = get_cluster_centers(x, best_clusters)
    return best_clusters, cluster_centers


def gradient_descent(x_name, func, func_dict, lower_bound, upper_bound, rhs_goal, max_iter=30, tolerance=1.0,
                     momentum=0.5, learning_rate=0.1):
    epsilon = 1e-4
    x = lower_bound + epsilon
    best_x = x
    best_delta = upper_bound - lower_bound
    counter = 0
    while True:
        func_dict[x_name] = x
        delta = func(**func_dict) - rhs_goal
        if abs(delta) < abs(best_delta):
            best_x = x
            best_delta = delta

        if abs(best_delta) < tolerance or counter >= max_iter:
            break

        x = x - learning_rate * delta

    return best_x


def spectral(x, k, frame_ids=None):
    clustering = SpectralClustering(n_clusters=k, assign_labels="discretize", affinity='cosine').fit(x)

    # find centers
    cluster_centers, best_ids = get_cluster_centers(x, clustering.labels_)
    return clustering.labels_, cluster_centers


def modularity_maximization_06_coframe_penalty_045_time995(x, k, frame_ids=None):
    similarity_graph = initialize_similarity_matrix(x, frame_ids)

    graph = EntityGraph()
    sample_ids = range(x.shape[0])
    keep_edge_percentage = 0.6
    sample_importance = np.ones(x.shape[0])
    graph.build_graph_entities(similarity_matrix=similarity_graph, entities=sample_ids,
                               entities_frequencies=sample_importance, keep_edge_percentage=keep_edge_percentage)
    partitions = graph.get_communities(time_resolution=.995)
    label_pred = np.asarray([v for k, v in partitions.items()])

    # find centers
    cluster_centers, best_ids = get_cluster_centers(x, label_pred)
    return label_pred, cluster_centers


def modularity_maximization_at_k(x, k, frame_ids=None):
    similarity_graph = initialize_similarity_matrix(x, frame_ids)

    graph = EntityGraph()
    sample_ids = range(x.shape[0])
    keep_edge_percentage = 0.6
    sample_importance = np.ones(x.shape[0])
    graph.build_graph_entities(similarity_matrix=similarity_graph, entities=sample_ids,
                               entities_frequencies=sample_importance, keep_edge_percentage=keep_edge_percentage)
    partitions = graph.get_communities_at_level(resolution=0.995, level=k)
    label_pred = np.asarray([v for k, v in partitions.items()])

    # find centers
    cluster_centers, best_ids = get_cluster_centers(x, label_pred)
    return label_pred, cluster_centers


def mean_shift(X, k, frame_ids=None):
    clustering = MeanShift().fit(X)

    # find centers
    cluster_centers, best_ids = get_cluster_centers(X, clustering.labels_)
    return clustering.labels_, cluster_centers


def mean_shift_orphans(x, k, frame_ids=None):
    """Not all data points assigned to clusters - some are disregarded as noise"""
    clustering = MeanShift(cluster_all=False).fit(x)

    # find centers
    cluster_centers, best_ids = get_cluster_centers(x, clustering.labels_)
    return clustering.labels_, cluster_centers


def agglomerative_at_k_complete(x, k, frame_ids=None):
    clustering = AgglomerativeClustering(n_clusters=k, compute_full_tree=True, affinity="cosine", linkage="complete").fit(x)

    # find centers
    cluster_centers, best_ids = get_cluster_centers(x, clustering.labels_)
    return clustering.labels_, cluster_centers


def agglomerative_at_k(x, k, frame_ids=None):
    clustering = AgglomerativeClustering(n_clusters=k, compute_full_tree=True, affinity="euclidean", linkage="average").fit(x)

    # find centers
    cluster_centers, best_ids = get_cluster_centers(x, clustering.labels_)
    return clustering.labels_, cluster_centers


def agglomerative(x, k, frame_ids=None):
    clustering = AgglomerativeClustering(compute_full_tree=True, affinity="cosine", linkage="complete").fit(x)

    # find centers
    cluster_centers, best_ids = get_cluster_centers(x, clustering.labels_)
    return clustering.labels_, cluster_centers


def get_cluster_centers(x, cluster_predictions):
    best_ids = []
    unique_clusters = set([c for c in cluster_predictions if c >= 0])
    cluster_centers = np.zeros((len(unique_clusters), x.shape[1]))
    if len(unique_clusters) == len(cluster_predictions):
        return x, cluster_predictions

    for cluster_id in unique_clusters:
        current_cluster_elements = x[cluster_predictions == cluster_id, :]
        cluster_size = current_cluster_elements.shape[0]

        current_cluster_center = np.mean(current_cluster_elements, axis=0)

        l2_norm_with_center = LA.norm(
            np.repeat([current_cluster_center], cluster_size, axis=0) - current_cluster_elements, axis=1)
        closest_to_center_idx = np.argmin(l2_norm_with_center)
        best_ids.append(closest_to_center_idx)
        cluster_centers[cluster_id] = current_cluster_center
    return cluster_centers, best_ids


def initialize_similarity_matrix(x, frame_ids):
    similarity_graph = cosine_similarity(x)

    # bboxes in the same frame are less likely to appear in the same keyframe
    if frame_ids:
        same_frame_penalty_factor = 0.45
        same_frame_vector = np.array([
            (sample_id_i[0], sample_id_j[0])
            for frame_id, sample_ids in groupby(enumerate(frame_ids), key=lambda item: item[1])
            for sample_id_j in sample_ids
            for sample_id_i in sample_ids
            if sample_id_i[0] != sample_id_j[0]])

        if len(same_frame_vector) > 0:
            adjusted_similarity_graph = similarity_graph.copy()

            for from_index, to_index in same_frame_vector:
                adjusted_value = similarity_graph[from_index, to_index] * same_frame_penalty_factor
                adjusted_similarity_graph[from_index, to_index] = adjusted_value
                adjusted_similarity_graph[to_index, from_index] = adjusted_value
            return adjusted_similarity_graph

        return similarity_graph


def estimate_clusters_significance(clusters, features, k, bbox_ids, confidences):
    dict_conf = dict(zip(bbox_ids, confidences))

    # clusters = get_clusters(features, False, k, bbox_ids)
    add_scores_to_clusters(clusters, dict_conf, features, bbox_ids)
    new_clusters = filter_clusters(clusters, features, bbox_ids, dict_conf)
    add_scores_to_clusters(new_clusters, dict_conf, features, bbox_ids)
    return new_clusters


def get_exp_dist(dist):
    sigma = 1.5
    return math.exp(-(dist / sigma)**2)


def get_score(conf, dist):
    return math.sqrt(conf * get_exp_dist(dist))


def add_dist2ct_per_key(clusters, features, image_ids, key):
    path_images = clusters[key]['path_images']
    indexes = [image_ids.index(path) for path in path_images]
    cluster_features = [features[i] for i in indexes]
    ct = np.mean(cluster_features, axis=0)
    dist2ct = [euclidean_distances(f[np.newaxis, :], ct[np.newaxis, :]) for f in cluster_features]
    dist2ct = [np.squeeze(d) for d in dist2ct]
    clusters[key]['dist2ct'] = dist2ct


def add_confidence_per_key(clusters, dict_conf, key):
    path_images = clusters[key]['path_images']
    conf = [dict_conf[p] for p in path_images]
    clusters[key]['conf'] = conf


def add_score_per_key(clusters, dict_conf, features, image_ids, key):
    add_dist2ct_per_key(clusters, features, image_ids, key)
    add_confidence_per_key(clusters, dict_conf, key)
    scores = [get_score(c, d) for d, c in zip(clusters[key]['dist2ct'], clusters[key]['conf'])]
    scores = [math.sqrt(s) for s in scores]
    clusters[key]['scores'] = scores


def add_scores_to_clusters(clusters, dict_conf, features, image_ids):
    print('\tAdding Score')
    for key in clusters.keys():
        add_score_per_key(clusters, dict_conf, features, image_ids, key)


def get_clusters(x, normalize_features, k, image_ids):
    min_norm = 0.01
    if normalize_features:
        x = [x / np.max([min_norm, np.linalg.norm(x)]) for x in x]
    x = np.asarray(x)

    # KMeans clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    cluster_classes = kmeans.predict(x)

    clusters = {}
    for key, val in zip(cluster_classes, image_ids):
        if key not in clusters:
            clusters[key] = {'path_images': []}
        clusters[key]['path_images'].append(val)

    return clusters


def filter_clusters(clusters, features, image_ids, dict_conf):
    print('\tFiltering Clusters')
    new_clusters = {}
    for key in clusters.keys():
        new_clusters[key] = {'path_images': clusters[key]['path_images']}
        new_clusters['outliers_' + str(key)] = {'path_images': []}
        while True:
            add_score_per_key(new_clusters, dict_conf, features, image_ids, key)
            scores = new_clusters[key]['scores']
            q25 = np.percentile(scores, 25)
            q75 = np.percentile(scores, 75)
            iqr = q75 - q25
            thresh = q25 - 1.5*iqr
            idx_to_discard = [i for i, s in enumerate(scores) if s < thresh]
            if len(idx_to_discard) == 0:
                break
            path_images = new_clusters[key]['path_images']
            new_clusters['outliers_' + str(key)]['path_images'] += [path for i, path in enumerate(path_images) if i in idx_to_discard]
            new_clusters[key]['path_images'] = [path for i, path in enumerate(path_images) if i not in idx_to_discard]
        if len(new_clusters['outliers_' + str(key)]['path_images']) == 0:
            new_clusters.pop('outliers_' + str(key))
    return new_clusters
