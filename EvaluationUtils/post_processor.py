import os
import shutil
from typing import Tuple

import pandas as pd
from EvaluationUtils.descriptive_stats import create_collage
from Utils.union_find import UnionFind
from Animator.consolidation_api import CharacterDetectionOutput
import numpy as np
from numpy import linalg as LA
from collections import Counter
from Animator.cluster_logic import ConsolidationEvaluator
import Animator.clustering_methods as cm
from .detection_mapping import DetectionMapping
from Animator.utils import recreate_dir, create_dir_if_not_exist
from sklearn.metrics.pairwise import cosine_similarity


class MockCharacterGrouper(object):
    def __init__(self, detected_bboxes, min_cluster_significance, keep_cluster_percentile, min_cluster_size,
                 dbscan_cluster_labels):
        """Sets the file names for data on entities"""
        features = np.asarray([detected_bbox.Features for detected_bbox in detected_bboxes])
        self.KeyFrameIndices = [detected_bbox.KeyFrameIndex for detected_bbox in detected_bboxes]
        self.FEATURES = features
        self.IDS = np.asarray([detected_bbox.Id for detected_bbox in detected_bboxes])
        self.DetectionConfidence = np.asarray([detected_bbox.Confidence for detected_bbox in detected_bboxes])
        self.DetectionThumbnailId = np.asarray([detected_bbox.ThumbnailId for detected_bbox in detected_bboxes])
        self.MinClusterSignificance = min_cluster_significance
        self.MinClusterSize = min_cluster_size
        self.KeepClusterPercentile = keep_cluster_percentile
        self.DbscanClusteringLabels = dbscan_cluster_labels

    @property
    def post_process_cluster_significance(self):
        # handle small input
        actual_input_size = self.FEATURES.shape[0]

        # load clustered data
        cluster_labels = self.DbscanClusteringLabels
        k_estimate = len(set([c for c in cluster_labels if c >= 0]))
        print('K estimated to be {} for {} bboxes'.format(k_estimate, actual_input_size))

        # find best thumbnail per cluster and cluster significance
        best_bbox_ids, cluster_significance, id_to_cluster_label, id_to_sample_significance, cluster_centers = \
            self.get_cluster_centers_best_bbox_and_significance(cluster_labels)

        # filter insignificant clusters (using negative cluster ids)
        cluster_labels, actual_best_k, best_bbox_ids, cluster_significance = \
            self.filter_insignificant_clusters(id_to_cluster_label, best_bbox_ids, cluster_significance)

        # filter insignificant samples per cluster
        cluster_labels, best_bbox_ids, actual_best_k = self.filter_insignificant_samples_per_cluster(cluster_labels)

        # keep cluster percentile
        cluster_labels, best_bbox_ids, actual_best_k = self.keep_cluster_percentile(cluster_labels, best_bbox_ids)

        # find best thumbnail per cluster and cluster significance
        best_bbox_ids, cluster_significance, id_to_cluster_label, id_to_sample_significance, cluster_centers = \
            self.get_cluster_centers_best_bbox_and_significance(cluster_labels)

        # filter insignificant clusters (using negative cluster ids)
        cluster_labels, actual_best_k, best_bbox_ids, cluster_significance = \
            self.filter_insignificant_clusters(id_to_cluster_label, best_bbox_ids, cluster_significance)

        # re-assign best thumbnail
        best_bbox_ids, cluster_significance, id_to_cluster_label, id_to_sample_significance, cluster_centers = \
            self.get_cluster_centers_best_bbox_and_significance(cluster_labels)

        print(f'[STATS]#5: Filtered insignificant clusters: {k_estimate-actual_best_k}')

        # get and merge the potential over-segmented clusters
        cluster_ids_to_merge, _ = self.should_consolidate_clusters(cluster_labels)
        cluster_labels, actual_best_k = MockCharacterGrouper.merge_clusters(cluster_ids_to_merge, cluster_labels)

        # re-assign best thumbnail
        best_bbox_ids, cluster_significance, id_to_cluster_label, id_to_sample_significance, cluster_centers = \
            self.get_cluster_centers_best_bbox_and_significance(cluster_labels)

        # print silhouette index
        ConsolidationEvaluator.unsupervised_evaluate_clusters(self.FEATURES, cluster_labels, 'OPTICS_ReCluster')

        return self.IDS, cluster_labels, actual_best_k, best_bbox_ids, cluster_significance, id_to_sample_significance

    def get_cluster_centers_best_bbox_and_significance(self, cluster_predictions, cluster_centers=None):
        best_thumbnails = []
        id_to_cluster_significance = dict()
        id_to_cluster_label = dict()
        id_to_sample_significance = dict()
        if not cluster_centers:
            cluster_centers = dict()
        for cluster_id in set(cluster_predictions):
            current_cluster_elements = self.FEATURES[cluster_predictions == cluster_id, :]
            cluster_size = current_cluster_elements.shape[0]
            current_cluster_bbox_ids = self.IDS[cluster_predictions == cluster_id]

            # ignore noise clusters
            if cluster_id >= 0 and cluster_size >= self.MinClusterSize:
                cluster_centers[cluster_id] = np.median(current_cluster_elements, axis=0) \
                    if cluster_centers is None or len(cluster_centers) == 0 or cluster_id not in cluster_centers \
                    else cluster_centers[cluster_id]
                cluster_detection_confidences = self.DetectionConfidence[cluster_predictions == cluster_id]

                # calculate cluster significance
                distance_from_center, closest_to_center_idx = self.calculate_distances_from_cluster_center(
                    cluster_centers[cluster_id], cluster_size, current_cluster_elements)
                best_thumbnails.append(current_cluster_bbox_ids[closest_to_center_idx])
                for d, c, bbox_id in zip(distance_from_center, cluster_detection_confidences, current_cluster_bbox_ids):
                    id_to_sample_significance[bbox_id] = cm.get_score(c, d)

                sig_scores = np.asarray(
                    [id_to_sample_significance[bbox_id_current_cluster] for bbox_id_current_cluster in
                     current_cluster_bbox_ids])
                cluster_significance = np.median(sig_scores)
            else:
                cluster_significance = 0.0

            for bbox_id in current_cluster_bbox_ids:
                id_to_cluster_significance[bbox_id] = cluster_significance
                id_to_cluster_label[bbox_id] = cluster_id

        return best_thumbnails, id_to_cluster_significance, id_to_cluster_label, id_to_sample_significance, cluster_centers

    def filter_insignificant_clusters(self, id_to_cluster_label, best_bbox_ids, cluster_significance):
        cluster_labels = []
        insignificant_cluster_id = -1
        filtered_cluster_ids = set()
        filtered_clusters_confidences = set()

        label_to_cluster_size = Counter(id_to_cluster_label.values())

        for bbox_prediction_id in self.IDS:
            # in case of significant cluster do nothing
            if cluster_significance[bbox_prediction_id] >= self.MinClusterSignificance and \
                    label_to_cluster_size[id_to_cluster_label[bbox_prediction_id]] >= self.MinClusterSize:
                cluster_labels.append(id_to_cluster_label[bbox_prediction_id])
                continue

            # update insignificant cluster
            if bbox_prediction_id in best_bbox_ids:
                best_bbox_ids.remove(bbox_prediction_id)

            # keep stats for telemetry
            filtered_cluster_ids.add(id_to_cluster_label[bbox_prediction_id])
            filtered_clusters_confidences.add(cluster_significance[bbox_prediction_id])

            id_to_cluster_label[bbox_prediction_id] = insignificant_cluster_id
            cluster_labels.append(insignificant_cluster_id)
            insignificant_cluster_id -= 1

        if len(filtered_cluster_ids) > 0:
            min_conf = min(filtered_clusters_confidences)
            max_conf = max(filtered_clusters_confidences)
            print(f'Filtered {len(filtered_cluster_ids)} clusters due to a score of range[{min_conf}, {max_conf}]')
        actual_best_k = len(set([cl for bbox_id, cl in id_to_cluster_label.items() if cl >= 0]))
        return np.asarray(cluster_labels), actual_best_k, best_bbox_ids, cluster_significance

    @staticmethod
    def calculate_distances_from_cluster_center(current_cluster_center, cluster_size, current_cluster_elements):
        if cluster_size <= 1:
            return np.asarray([0.0]), np.asarray([0])
        l2_norm_with_center = LA.norm(
            np.repeat([current_cluster_center], cluster_size, axis=0) - current_cluster_elements, axis=1)
        closest_to_center_idx = np.argmin(l2_norm_with_center)
        fares_from_center_idx = np.argmax(l2_norm_with_center)

        # calculate cluster significance
        distance_from_center = l2_norm_with_center / l2_norm_with_center[fares_from_center_idx]
        return distance_from_center, closest_to_center_idx

    def filter_insignificant_samples_per_cluster(self, cluster_labels):
        best_bbox_ids = []
        bbox_id_to_index = dict(zip(self.IDS, range(len(self.IDS))))
        smallest_label = min(cluster_labels)
        insignificant_cluster_label = -1 if smallest_label >= 0 else smallest_label - 1

        for cluster_id in set(cluster_labels):
            while True:
                current_cluster_elements = self.FEATURES[cluster_labels == cluster_id, :]
                cluster_size = current_cluster_elements.shape[0]
                current_cluster_bbox_ids = self.IDS[cluster_labels == cluster_id]
                if cluster_size < self.MinClusterSize:
                    # regard outliers as noise
                    for id_to_discard in current_cluster_bbox_ids:
                        cluster_labels[bbox_id_to_index[id_to_discard]] = insignificant_cluster_label
                        insignificant_cluster_label -= 1
                    break

                current_cluster_center = np.median(current_cluster_elements, axis=0)
                cluster_detection_confidences = self.DetectionConfidence[cluster_labels == cluster_id]

                distance_from_center, closest_to_center_idx = self.calculate_distances_from_cluster_center(
                    current_cluster_center, cluster_size, current_cluster_elements)

                id_to_sample_significance = dict()
                for d, c, bbox_id in zip(distance_from_center, cluster_detection_confidences, current_cluster_bbox_ids):
                    id_to_sample_significance[bbox_id] = cm.get_score(c, d)

                sig_scores = list(id_to_sample_significance.values())
                q25 = np.percentile(sig_scores, 25)
                q75 = np.percentile(sig_scores, 75)
                iqr = q75 - q25
                thresh = max(q25 - 1. * iqr, 0)

                ids_to_discard = [bbox_id for bbox_id in current_cluster_bbox_ids
                                  if id_to_sample_significance[bbox_id] < thresh]

                if len(ids_to_discard) == 0:
                    best_bbox_ids.append(closest_to_center_idx)
                    break

                # regard outliers as noise
                for id_to_discard in ids_to_discard:
                    cluster_labels[bbox_id_to_index[id_to_discard]] = insignificant_cluster_label
                    insignificant_cluster_label -= 1
                    discarded_thumbnailid = self.DetectionThumbnailId[bbox_id_to_index[id_to_discard]]
                    print(f'Filtered out bbox_id: {id_to_discard} with thumbnail: {discarded_thumbnailid} since it is'
                          f' an outlier of cluster: {cluster_id} with significance (or distance score from center) of: '
                          f'{id_to_sample_significance[id_to_discard]:.4f} where the threshold was: {thresh:.4f}')

        actual_best_k = len(set(lab for lab in cluster_labels if lab >= 0))
        return cluster_labels, best_bbox_ids, actual_best_k

    def keep_cluster_percentile(self, cluster_labels, best_bbox_ids):
        if self.KeepClusterPercentile == 1.0:
            print('KeepClusterPercentile is 100%: No filtering by KeepClusterPercentile...')
            return cluster_labels, best_bbox_ids, len(set(lab for lab in cluster_labels if lab >= 0))

        smallest_label = min(cluster_labels)
        insignificant_cluster_label = -1 if smallest_label >= 0 else smallest_label - 1
        filtered_cluster_ids = set()

        id_to_cluster_label = dict(zip(self.IDS, cluster_labels))
        label_to_cluster_size = Counter(id_to_cluster_label.values())
        valid_cluster_stats = sorted([{'label': lab, 'size': cluster_size}
                                      for lab, cluster_size in label_to_cluster_size.items()
                                      if lab >= 0], key=lambda cs: cs['size'], reverse=True)
        new_cluster_labels = []
        total_valid_points = sum([vcs['size'] for vcs in valid_cluster_stats])
        cumsum_buffer = 0.
        for vcs in valid_cluster_stats:
            vcs['percentage'] = 1. * vcs['size'] / total_valid_points
            vcs['cumsum'] = vcs['percentage'] + cumsum_buffer
            vcs['is_valid'] = vcs['cumsum'] <= self.KeepClusterPercentile or len(valid_cluster_stats) <= 5
            cumsum_buffer += vcs['percentage']

        label_to_validity = dict([(vcs['label'], vcs['is_valid']) for vcs in valid_cluster_stats])

        for bbox_prediction_id in self.IDS:
            # in case of significant cluster do nothing
            if id_to_cluster_label[bbox_prediction_id] < 0 or \
                    label_to_validity[id_to_cluster_label[bbox_prediction_id]]:
                new_cluster_labels.append(int(id_to_cluster_label[bbox_prediction_id]))
                continue

            # update insignificant cluster
            if bbox_prediction_id in best_bbox_ids:
                best_bbox_ids.remove(bbox_prediction_id)

            # keep stats for telemetry
            filtered_cluster_ids.add(id_to_cluster_label[bbox_prediction_id])

            id_to_cluster_label[bbox_prediction_id] = insignificant_cluster_label
            new_cluster_labels.append(int(insignificant_cluster_label))
            insignificant_cluster_label -= 1

        n_filtered_clusters = len(filtered_cluster_ids)
        if n_filtered_clusters > 0:
            if n_filtered_clusters > 20:
                print(f'Filtered {n_filtered_clusters} clusters due to percentile cutoff.')
            else:
                print('The following clusters were filtered due to percentile cutoff: {}'.format(filtered_cluster_ids))
        actual_best_k = len(set([cl for bbox_id, cl in id_to_cluster_label.items() if cl >= 0]))
        return np.asarray(new_cluster_labels), best_bbox_ids, actual_best_k

    def should_consolidate_clusters(self, cluster_predictions):
        """compute the clusters similarity for post-processing merge"""
        valid_clusters = sorted(cid for cid in set(cluster_predictions) if cid >= 0)
        n = len(valid_clusters)
        if n <= 1:
            raise Exception('Something went wrong... Found a single (or no) cluster/s!')
        cluster_sim = np.zeros([n, n], dtype=float)
        features = self.FEATURES[cluster_predictions >= 0, :]
        cluster_ids = cluster_predictions[cluster_predictions >= 0]
        partition = dict()
        for i in range(len(cluster_ids)):
            partition[cluster_ids[i]] = partition.get(cluster_ids[i], set()) | {i}

        for i in range(n):
            left_cluster_id = valid_clusters[i]
            left_cluster_feats = features[cluster_ids == left_cluster_id, :]
            for j in range(i+1, n):
                right_cluster_id = valid_clusters[j]
                right_cluster_feats = features[cluster_ids == right_cluster_id, :]

                cosine_sim = cosine_similarity(left_cluster_feats, right_cluster_feats)
                cluster_sim[i, j] = cosine_sim.mean()
                cluster_sim[j, i] = cosine_sim.mean()

        m_merges = n // 2
        top_3 = largest_indices(cluster_sim, 2*m_merges)
        couples_indices = []
        couples_sims = []
        top_percentile_cluster_distance = max(.63, min(.69, np.quantile(cluster_sim.flatten(), 0.975)))
        print(f'Using the 0.6 <= 99th percentile <= 0.7 as the merging cutoff: {top_percentile_cluster_distance:.4f}')
        for i in range(m_merges):
            left_index = top_3[0][2*i]
            right_index = top_3[0][2*i+1]
            clusters_cosine_similarity = cluster_sim[left_index, right_index]

            # take only cluster sim of more than the 99th percentile
            if clusters_cosine_similarity < top_percentile_cluster_distance:
                print(f'Skipping the merge of cluster: {valid_clusters[left_index]} with cluster:'
                      f' {valid_clusters[right_index]} due to similarity of: '
                      f'{clusters_cosine_similarity:.4f} < {top_percentile_cluster_distance:.4f}')
                continue

            couples_indices.append((valid_clusters[left_index], valid_clusters[right_index]))
            couples_sims.append(clusters_cosine_similarity)
        print(f'The top {m_merges} coupled clusters are {couples_indices} with similarities: {couples_sims}')
        print(f'[STATS]#5.5: Num consolidated clusters: {len(couples_indices)}')
        return couples_indices, couples_sims

    @staticmethod
    def merge_clusters(cluster_ids_to_merge: list, cluster_labels: np.ndarray) -> Tuple[np.ndarray, int]:
        """update the predicted clusters according to the groups by cluster similarity"""
        if len(cluster_ids_to_merge) == 0:
            valid_clusters = set(c for c in cluster_labels if c >= 0)
            return cluster_labels, len(valid_clusters)

        grouped_cluster_ids = UnionFind.disjoint_sets(cluster_ids_to_merge)
        cluster_to_rep = dict()
        for c_group in grouped_cluster_ids:
            rep = min(c_group)
            for cluster_id in c_group:
                cluster_to_rep[cluster_id] = rep
        reassigned_cluster_labels = cluster_labels.copy()
        for i in range(cluster_labels.shape[0]):
            reassigned_cluster_labels[i] = cluster_to_rep.get(cluster_labels[i], cluster_labels[i])
        valid_grouped_clusters = set(c for c in reassigned_cluster_labels if c >= 0)
        return reassigned_cluster_labels, len(valid_grouped_clusters)


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def post_process_single_episode(eval_root, ser, role):
    _min_cluster_sig = 0.725
    # _keep_cluster_percentile = .95
    _keep_cluster_percentile = .975
    _min_cluster_size = 3
    ser_path = os.path.join(eval_root, '', ser)
    role_path = os.path.join(ser_path, '', role)
    detection_output_path = os.path.join(role_path, 'animationdetectionoutput.json')
    print('Series: {}, Role: {}'.format(ser, role))

    character_detections = CharacterDetectionOutput.read_from_json(detection_output_path)
    grouping_output_path = os.path.join(role_path, 'animationgroupingoutput.json')
    mapping = DetectionMapping.parse_index(detection_output_path, grouping_output_path)
    id_to_group = {mapp.Id: mapp.BoxesConsolidation for mapp in mapping}

    detection_id_set = set(d.ThumbnailId for d in character_detections.CharacterBoundingBoxes)
    grouping_id_set = set(m.ThumbnailId for m in mapping)
    xor_groups = (detection_id_set - grouping_id_set) | (grouping_id_set - detection_id_set)
    if len(xor_groups) > 0:
        print('The following ids are a mismatch between detection and grouping:\n{}. SKIPPING THEM!\n' \
              .format(xor_groups))
        character_detections.CharacterBoundingBoxes = \
            filter(lambda detection: detection.ThumbnailId not in xor_groups,
                   character_detections.CharacterBoundingBoxes)
    cluster_labels = np.asarray([id_to_group[d.Id]
                                 for d in character_detections.CharacterBoundingBoxes if d.Id in id_to_group])
    k_recluster = len(set([c for c in cluster_labels if c >= 0]))
    print('DBSCAN k={}'.format(k_recluster))
    mock_grouper = MockCharacterGrouper(character_detections.CharacterBoundingBoxes, _min_cluster_sig,
                                        _keep_cluster_percentile, _min_cluster_size, cluster_labels)
    ids, cluster_labels, final_k, best_bbox_ids, _cluster_significance, _id_to_sample_significance = \
        mock_grouper.post_process_cluster_significance
    id_to_cluster_label = dict(zip(ids, cluster_labels))
    print('Results: k={}'.format(final_k))
    cluster_label_to_sig = {
        id_to_cluster_label[post_id]: _cluster_significance[post_id]
        if post_id in _cluster_significance
        else 0.
        for post_id in ids
        if id_to_cluster_label[post_id] >= 0
    }
    ordered_cluster_significances = sorted(cluster_label_to_sig.items(), key=lambda tup: tup[1],
                                           reverse=True)
    print(''.join(['ClusterId:{} -> Sig:{:.4f}\n'.format(t[0], t[1]) for t in ordered_cluster_significances]))

    # copy significant clusters collage
    significant_collage_repo = recreate_dir(role_path, 'significant_collages')
    groups_repo = recreate_dir(role_path, 'groups')

    source_detections_repo = os.path.join(role_path, 'animationdetectionoriginalimages')
    for cid, sig in ordered_cluster_significances:
        cluster_bbox_ids = ids[cluster_labels == cid]
        cluster_collage_thumbnail_ids = [detection.ThumbnailId for detection in
                                         character_detections.CharacterBoundingBoxes if
                                         detection.Id in cluster_bbox_ids]
        collage_images = [os.path.join(source_detections_repo, '{}.jpg'.format(bbox_thumb_id))
                          for bbox_thumb_id in cluster_collage_thumbnail_ids]
        cluster_collage_name = 'Cluster_{}Sig_{:.4f}'.format(cid, sig)
        target_collage_path = os.path.join(significant_collage_repo, '{}.jpg'.format(cluster_collage_name))
        create_collage(collage_images, target_collage_path)
        cluster_group_repo = create_dir_if_not_exist(groups_repo, f'cluster_{cid}')
        for source_det_im_path in collage_images:
            det_file_name = os.path.basename(source_det_im_path)
            dest_det_path = os.path.join(cluster_group_repo, det_file_name)
            shutil.copy(source_det_im_path, dest_det_path)

    # keep predictions in a dataframe
    num_bboxes = sum(1. for c in cluster_labels if c >= 0)
    avg_bbox_per_cluster = num_bboxes / final_k
    pred_row = dict(NumProposals_1=len(list(character_detections.CharacterBoundingBoxes)), InitialK_2=-1,
                    DbscanK_3=-1, ReClusterK_4=k_recluster, DicardedK_5=k_recluster - final_k, FinalK_6=final_k,
                    AvgNumProposalsPerCluster=avg_bbox_per_cluster, SeriesName=ser, Role=role, ValidBoxes=num_bboxes)
    print(f'[STATS]#6: Final N clusters: {len(ordered_cluster_significances)}')
    print(f'[STATS]#7: Num boxes per cluster: {num_bboxes/len(ordered_cluster_significances)}')
    return pred_row


def post_processor_main():
    eval_root = r'..\TripletsSeNet'
    stats_df_path = r'..\TripletsSeNet\GroupingStats\ClusteringStats.tsv'

    predictions_df = pd.DataFrame({'SeriesName': [], 'Role': [], 'NumProposals_1': [], 'InitialK_2': [],
                                   'DbscanK_3': [], 'ReClusterK_4': [], 'DicardedK_5': [], 'FinalK_6': [],
                                   'AvgNumProposalsPerCluster': []})
    series = [s for s in os.listdir(eval_root) if s not in ['Transformers', 'DextersLab', 'Cars'] and os.path.isdir(s)]
    for ser in series:
        for role in ['Training']:
            pred_row = post_process_single_episode(eval_root, ser, role)
            predictions_df = predictions_df.append(pred_row, ignore_index=True)

            print('Finished an episode...')
    predictions_df.to_csv(stats_df_path, header=True, sep='\t')
    return


# if __name__ == '__main__':
#     post_processor_main()
#     print('Done!')
