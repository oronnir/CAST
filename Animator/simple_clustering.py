import math
import os
from copy import deepcopy

import numpy as np

from Animator.clustering_methods import k_means, c_dbscan, c_optics
from Animator.cluster_logic import CharacterConsolidator
from Animator.cluster_logic import ConsolidationEvaluator


def my_simple_clustering(ids, features, frame_ids):
    """
    Binary search for the right eps parameter by DBSCAN
    :param ids: the box ids
    :param features: the feature vector per box
    :param frame_ids: the frame ids/indices
    :return: ids, cluster_ids, k_estimate, best_thumbnails
    """
    min_input_size = 50
    actual_input_size = features.shape[0]
    if actual_input_size <= min_input_size:
        print('\nVideo contains less than minimal number of bboxes ({}), minimum is {}'
              .format(features.shape[0], min_input_size))
        return ids, np.asarray(range(actual_input_size)), actual_input_size, features

    # cluster the data
    k_estimate = min(max(15, int(1. * actual_input_size / 40)), 60)
    k_estimate_ub = k_estimate
    k_estimate_lb = max(5, k_estimate // 4)
    print('K estimated to be {} for {} bboxes'.format(k_estimate, actual_input_size))

    best_score = -1
    best_silhouette = -1
    best_p_noise = 1
    best_cluster_ids = None
    best_centers = None
    best_k_estimate = -1
    cluster_ids = None
    cluster_centers = None
    score_per_k = []
    for k in range(k_estimate_lb, k_estimate_ub):
        try:
            cluster_ids, cluster_centers = c_dbscan(features, k, frame_ids, ids, eps_span=15, initial_eps=1e-3)
            silhouette, percentage_noise = ConsolidationEvaluator.unsupervised_evaluate_clusters(features, cluster_ids,
                                                                                            'c_dbscan')
            p_noise = percentage_noise/100
            validity_and_silhouette_score = silhouette*(1-p_noise)
            score_per_k.append({'k': k, 'silhouette': silhouette, 'p_noise': p_noise,
                                'score': validity_and_silhouette_score})
        except Exception as e:
            print(f'Failed on k={k} with exception {e}')
            silhouette = -1
            p_noise = 1
            validity_and_silhouette_score = -1
        if validity_and_silhouette_score > best_score:
            print(f'*** for k={k} -> silhouette={silhouette}, p_noise={p_noise} ***')
            best_score = validity_and_silhouette_score
            best_silhouette = silhouette
            best_p_noise = p_noise
            best_cluster_ids = cluster_ids
            best_centers = cluster_centers
            best_k_estimate = k
    cluster_ids = best_cluster_ids
    cluster_centers = best_centers
    actual_k = len(set([cid for cid in cluster_ids if cid >= 0]))
    print(f'[STATS]#2: The opt k estimate is {best_k_estimate} with Silhouette: {best_silhouette:.4f}, '
          f'p_noise: {best_p_noise:.4f}, and [STATS]#3: the actual DBSCAN n_clusters: {actual_k}')
    print(f'The k search full log is:{os.linesep}{os.linesep.join([str(elem) for elem in score_per_k])}')

    cluster_ids, cluster_centers = re_cluster_noisy_samples(features, ids, cluster_ids, cluster_centers, frame_ids)
    ConsolidationEvaluator.unsupervised_evaluate_clusters(features, cluster_ids, 'OPTICS re-cluster')

    best_thumbnails = get_cluster_centers_thumbnail_id(ids, features, cluster_ids, cluster_centers)
    actual_k_recluster = len(set([cid for cid in cluster_ids if cid >= 0]))
    print(f'[STATS]#4: Post re-cluster actual k: {actual_k_recluster}')
    return ids, cluster_ids, k_estimate, best_thumbnails


def get_cluster_centers_thumbnail_id(ids, features, cluster_predictions, cluster_centers):
    best_thumbnails = []
    id_to_cluster_label = dict()
    for cluster_id in set(cluster_predictions):
        current_cluster_elements = features[cluster_predictions == cluster_id, :]
        cluster_size = current_cluster_elements.shape[0]
        current_cluster_bbox_ids = ids[cluster_predictions == cluster_id]

        # ignore noise clusters
        if cluster_id >= 0 and cluster_size >= 5:
            current_cluster_center = np.median(current_cluster_elements, axis=0) \
                if cluster_centers is None or len(cluster_centers) == 0 \
                else cluster_centers[cluster_id]

            # calculate cluster significance
            distance_from_center, closest_to_center_idx = CharacterConsolidator.calculate_distances_from_cluster_center(
                current_cluster_center, cluster_size, current_cluster_elements)
            best_thumbnails.append(current_cluster_bbox_ids[closest_to_center_idx])

        for bbox_id in current_cluster_bbox_ids:
            id_to_cluster_label[bbox_id] = cluster_id

    return best_thumbnails


def re_cluster_large_clusters(features, ids, cluster_ids, cluster_centers, max_cluster_size):
    """re-cluster the large clusters which holds more than max_cluster_size samples"""
    original_order = np.asarray(range(len(ids)))
    next_available_cluster_id = max(cluster_ids) + 1
    id_to_cluster = {
        cid: {
            'Features': features[cluster_ids == cid, :],
            'BboxIds': ids[cluster_ids == cid],
            'ClusterCenter': cluster_centers[cid],
            'OriginalOrder': original_order[cluster_ids == cid],
            'ClusterId': cid
        }
        for cid in set(cluster_ids)
    }

    cloned_id_to_cluster = deepcopy(id_to_cluster)
    for cluster_id, _ in id_to_cluster.items():
        cluster_meta = id_to_cluster[cluster_id]

        # ignore noise
        if cluster_id < 0:
            continue

        # skip the valid clusters
        cluster_size = cluster_meta['Features'].shape[0]
        if cluster_size <= max_cluster_size:
            continue

        # re-cluster
        k_new = max(2, 1 + int(cluster_size/50))
        internal_cluster_ids, internal_cluster_centers = k_means(cluster_meta['Features'], k_new)

        # reassign cluster ids
        original_internal_cluster_ids = set(internal_cluster_ids)
        for internal_cluster_id in original_internal_cluster_ids:
            cloned_id_to_cluster[next_available_cluster_id] = {
                'Features': cluster_meta['Features'][internal_cluster_ids == internal_cluster_id, :],
                'BboxIds': cluster_meta['BboxIds'][internal_cluster_ids == internal_cluster_id],
                'ClusterCenter': internal_cluster_centers[internal_cluster_id],
                'OriginalOrder': cluster_meta['OriginalOrder'][internal_cluster_ids == internal_cluster_id],
                'ClusterId': next_available_cluster_id
            }

            next_available_cluster_id += 1
        next_available_cluster_id -= 1
        cloned_id_to_cluster[cluster_id] = cloned_id_to_cluster[next_available_cluster_id]
        cloned_id_to_cluster[cluster_id]['ClusterId'] = cluster_id
        cloned_id_to_cluster.pop(next_available_cluster_id)

    id_to_cluster = cloned_id_to_cluster

    cluster_ids_and_original_order = []
    for cluster in id_to_cluster.values():
        for bbox in cluster['OriginalOrder']:
            cluster_ids_and_original_order.append((cluster['ClusterId'], bbox))
    ordered_cluster_ids_by_original = sorted(cluster_ids_and_original_order, key=lambda t: t[1])
    cluster_ids = np.asarray([ocibo[0] for ocibo in ordered_cluster_ids_by_original])
    ordered_cluster_centers_by_cluster_id = [clust['ClusterCenter'] for clust in
                                             sorted([c for c in id_to_cluster.values()
                                                     if c['ClusterId'] >= 0], key=lambda t: t['ClusterId'])]
    cluster_centers = np.asarray(ordered_cluster_centers_by_cluster_id)
    print('re_cluster has finished with {} clusters'.format(len([k for k in id_to_cluster.keys() if k >= 0])))
    return cluster_ids, cluster_centers


def re_cluster_noisy_samples(features, ids, cluster_ids, cluster_centers, frame_ids):
    """run clustering again on the noisy samples which were discarded on the first attempt"""
    local_frame_ids = np.asarray(frame_ids)
    original_order = np.asarray(range(len(ids)))
    next_available_cluster_id = max(cluster_ids) + 1
    id_to_cluster = {
        cid: {
            'Features': features[cluster_ids == cid, :],
            'BboxIds': ids[cluster_ids == cid],
            'FrameIds': local_frame_ids[cluster_ids == cid],
            'ClusterCenter': cluster_centers[cid],
            'OriginalOrder': original_order[cluster_ids == cid],
            'ClusterId': cid
        }
        for cid in set(cluster_ids)
    }

    cloned_id_to_cluster = deepcopy(id_to_cluster)
    cluster_id = -1
    if cluster_id not in id_to_cluster:
        print('No noise samples in this dataset -> re_cluster_noisy_samples() abstains...')
        return cluster_ids, cluster_centers

    cluster_meta = id_to_cluster[cluster_id]

    # skip the valid clusters
    cluster_size = cluster_meta['Features'].shape[0]

    # re-cluster
    k_new = max(2, 1 + int(math.sqrt(cluster_size/50)))

    internal_cluster_ids, internal_cluster_centers = \
        c_optics(cluster_meta['Features'], k_new, cluster_meta['FrameIds'], cluster_meta['BboxIds'])

    # reassign cluster ids
    original_internal_cluster_ids = set(internal_cluster_ids)
    valid_internal_cluster_ids = [oici for oici in original_internal_cluster_ids if oici >= 0]
    if len(valid_internal_cluster_ids) <= 1:
        print('OPTICS solution is insignificant -> re_cluster_noisy_samples() abstains...')
        return cluster_ids, cluster_centers

    for internal_cluster_id in original_internal_cluster_ids:
        cloned_id_to_cluster[next_available_cluster_id] = {
            'Features': cluster_meta['Features'][internal_cluster_ids == internal_cluster_id, :],
            'BboxIds': cluster_meta['BboxIds'][internal_cluster_ids == internal_cluster_id],
            'ClusterCenter': internal_cluster_centers[internal_cluster_id],
            'OriginalOrder': cluster_meta['OriginalOrder'][internal_cluster_ids == internal_cluster_id],
            'ClusterId': next_available_cluster_id
        }

        next_available_cluster_id += 1
    next_available_cluster_id -= 1
    cloned_id_to_cluster[cluster_id] = cloned_id_to_cluster[next_available_cluster_id]
    cloned_id_to_cluster[cluster_id]['ClusterId'] = cluster_id
    cloned_id_to_cluster.pop(next_available_cluster_id)

    id_to_cluster = cloned_id_to_cluster

    cluster_ids_and_original_order = []
    for cluster in id_to_cluster.values():
        for bbox in cluster['OriginalOrder']:
            cluster_ids_and_original_order.append((cluster['ClusterId'], bbox))
    ordered_cluster_ids_by_original = sorted(cluster_ids_and_original_order, key=lambda t: t[1])
    cluster_ids = np.asarray([ocibo[0] for ocibo in ordered_cluster_ids_by_original])
    ordered_cluster_centers_by_cluster_id = [clust['ClusterCenter'] for clust in
                                             sorted([c for c in id_to_cluster.values()
                                                     if c['ClusterId'] >= 0], key=lambda t: t['ClusterId'])]
    cluster_centers = np.asarray(ordered_cluster_centers_by_cluster_id)
    print('re_cluster has finished with {} clusters'.format(len([k for k in id_to_cluster.keys() if k >= 0])))
    return cluster_ids, cluster_centers
