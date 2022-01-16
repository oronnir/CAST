from collections import Counter

import numpy as np
import sklearn.metrics
from numpy import linalg as LA
from sklearn import metrics

import Animator.clustering_methods as cm
import Animator.evaluation_metrics as em

RANDOM_SEED = 1234567
np.random.seed(RANDOM_SEED)


class CharacterGrouper(object):
    def __init__(self, detected_bboxes, cluster_analysis, normalization_method, min_cluster_significance, keep_cluster_percentile, min_cluster_size):
        """Sets the file names for data on entities"""
        features = np.asarray([detected_bbox.Features for detected_bbox in detected_bboxes])
        self.KeyFrameIndices = [detected_bbox.KeyFrameIndex for detected_bbox in detected_bboxes]
        self.FEATURES = np.nan_to_num(normalization_method(features))
        self.CLUSTER_ANALYSIS = cluster_analysis
        self.IDS = np.asarray([detected_bbox.Id for detected_bbox in detected_bboxes])
        self.DetectionConfidence = np.asarray([detected_bbox.Confidence for detected_bbox in detected_bboxes])
        self.DetectionThumbnailId = np.asarray([detected_bbox.ThumbnailId for detected_bbox in detected_bboxes])
        self.MinClusterSignificance = min_cluster_significance
        self.MinClusterSize = min_cluster_size
        self.KeepClusterPercentile = keep_cluster_percentile

    @property
    def group_by_features(self):
        # handle small input
        min_input_size = 30
        actual_input_size = self.FEATURES.shape[0]
        if actual_input_size <= min_input_size:
            print('\nVideo contains less than minimal number of bboxes ({}), minimum is {}'
                  .format(self.FEATURES.shape[0], min_input_size))
            return self.IDS, np.asarray(range(actual_input_size)), actual_input_size, self.IDS

        print('\nStart analyzing {} over a video with {} bounding boxes'
              .format(self.CLUSTER_ANALYSIS.__name__, self.FEATURES.shape[0]))

        # cluster the data
        k_estimate = min(max(10, int(1.*actual_input_size/10)), 75)
        print('K etimated to be {} for {} bboxes'.format(k_estimate, actual_input_size))
        cluster_labels, cluster_centers = self.CLUSTER_ANALYSIS(self.FEATURES, k_estimate, self.KeyFrameIndices)

        # find best thumbnail per cluster and cluster significance
        best_bbox_ids, cluster_significance, id_to_cluster_label, id_to_sample_significance = \
            self.get_cluster_centers_best_bbox_and_significance(cluster_labels, cluster_centers)

        # filter insignificant clusters (using negative cluster ids)
        cluster_labels, actual_best_k, best_bbox_ids, cluster_significance = \
            self.filter_insignificant_clusters(id_to_cluster_label, best_bbox_ids, cluster_significance)

        # filter insignificant samples per cluster
        cluster_labels, best_bbox_ids, actual_best_k = self.filter_insignificant_samples_per_cluster(cluster_labels)

        # keep cluster percentile
        cluster_labels, best_bbox_ids, actual_best_k = self.keep_cluster_percentile(cluster_labels, best_bbox_ids)

        # re-assign best thumbnail
        best_bbox_ids, cluster_significance, id_to_cluster_label, id_to_sample_significance = \
            self.get_cluster_centers_best_bbox_and_significance([cid for cid in cluster_labels])

        # print silhouette index
        GroupingEvaluator.unsupervised_evaluate_clusters(self.FEATURES, cluster_labels, self.CLUSTER_ANALYSIS.__name__)

        return self.IDS, cluster_labels, actual_best_k, best_bbox_ids

    def get_cluster_centers_best_bbox_and_significance(self, cluster_predictions, cluster_centers=None):
        best_thumbnails = []
        id_to_cluster_significance = dict()
        id_to_cluster_label = dict()
        id_to_sample_significance = dict()
        for cluster_id in set(cluster_predictions):
            current_cluster_elements = self.FEATURES[cluster_predictions == cluster_id, :]
            cluster_size = current_cluster_elements.shape[0]
            current_cluster_bbox_ids = self.IDS[cluster_predictions == cluster_id]

            # ignore noise clusters
            if cluster_id >= 0 and cluster_size >= self.MinClusterSize:
                current_cluster_center = np.median(current_cluster_elements, axis=0) \
                    if cluster_centers is None or len(cluster_centers) == 0 \
                    else cluster_centers[cluster_id]

                cluster_detection_confidences = self.DetectionConfidence[cluster_predictions == cluster_id]

                # calculate cluster significance
                distance_from_center, closest_to_center_idx = self.calculate_distances_from_cluster_center(
                    current_cluster_center, cluster_size, current_cluster_elements)
                best_thumbnails.append(current_cluster_bbox_ids[closest_to_center_idx])
                for d, c, bbox_id in zip(distance_from_center, cluster_detection_confidences, current_cluster_bbox_ids):
                    id_to_sample_significance[bbox_id] = cm.get_score(c, d)

                sig_scores = id_to_sample_significance.values()
                cluster_significance = np.median(sig_scores)
            else:
                cluster_significance = 0.0

            for bbox_id in current_cluster_bbox_ids:
                id_to_cluster_significance[bbox_id] = cluster_significance
                id_to_cluster_label[bbox_id] = cluster_id

        return best_thumbnails, id_to_cluster_significance, id_to_cluster_label, id_to_sample_significance

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
            min_confidence = min(filtered_clusters_confidences)
            max_confidence = max(filtered_clusters_confidences)
            print('The following clusters were filtered due to a score of range[{}, {}]: {}'
                  .format(min_confidence, max_confidence, filtered_cluster_ids))
        actual_best_k = len(set([cl for bbox_id, cl in id_to_cluster_label.items() if cl >= 0]))
        return cluster_labels, actual_best_k, best_bbox_ids, cluster_significance

    @staticmethod
    def calculate_distances_from_cluster_center(current_cluster_center, cluster_size, current_cluster_elements):
        if cluster_size <= 1:
            return np.asarray([0.0]), np.asarray([0])

        l2_norm_with_center = LA.norm(
            np.repeat([current_cluster_center], cluster_size, axis=0) - current_cluster_elements, axis=1)
        closest_to_center_idx = np.argmin(l2_norm_with_center)
        farrest_from_center_idx = np.argmax(l2_norm_with_center)

        # calculate cluster significance
        distance_from_center = l2_norm_with_center / l2_norm_with_center[farrest_from_center_idx]
        return distance_from_center, closest_to_center_idx

    def filter_insignificant_samples_per_cluster(self, cluster_labels):
        best_bbox_ids = []
        bbox_id_to_index = dict(zip(self.IDS, range(len(self.IDS))))
        smallest_label = min(cluster_labels)
        insignificant_cluster_label = -1 if smallest_label >= 0 else smallest_label-1

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

                sig_scores = id_to_sample_significance.values()
                q25 = np.percentile(sig_scores, 25)
                q75 = np.percentile(sig_scores, 75)
                iqr = q75 - q25
                thresh = max(q25 - .25 * iqr, 0)

                ids_to_discard = [bbox_id for bbox_id in current_cluster_bbox_ids
                                  if id_to_sample_significance[bbox_id] < thresh]

                if len(ids_to_discard) == 0:
                    best_bbox_ids.append(closest_to_center_idx)
                    break

                # regard outliers as noise
                for id_to_discard in ids_to_discard:
                    cluster_labels[bbox_id_to_index[id_to_discard]] = insignificant_cluster_label
                    insignificant_cluster_label -= 1
                    print('Filtered out bbox_id: {} since it is an outlier of cluster: {} with distance score from '
                          'center of: {} where the threshold was: {}'
                          .format(id_to_discard, cluster_id, id_to_sample_significance[id_to_discard], thresh))

        actual_best_k = len(set(cluster_labels))
        return cluster_labels, best_bbox_ids, actual_best_k

    def keep_cluster_percentile(self, cluster_labels, best_bbox_ids):
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
            vcs['percentage'] = 1.*vcs['size']/total_valid_points
            vcs['cumsum'] = vcs['percentage'] + cumsum_buffer
            vcs['is_valid'] = vcs['cumsum'] <= self.KeepClusterPercentile
            cumsum_buffer += vcs['percentage']

        label_to_validity = dict([(vcs['label'], vcs['is_valid']) for vcs in valid_cluster_stats])

        for bbox_prediction_id in self.IDS:
            # in case of significant cluster do nothing
            if id_to_cluster_label[bbox_prediction_id] < 0 or label_to_validity[id_to_cluster_label[bbox_prediction_id]]:
                new_cluster_labels.append(id_to_cluster_label[bbox_prediction_id])
                continue

            # update insignificant cluster
            if bbox_prediction_id in best_bbox_ids:
                best_bbox_ids.remove(bbox_prediction_id)

            # keep stats for telemetry
            filtered_cluster_ids.add(id_to_cluster_label[bbox_prediction_id])

            id_to_cluster_label[bbox_prediction_id] = insignificant_cluster_label
            new_cluster_labels.append(insignificant_cluster_label)
            insignificant_cluster_label -= 1

        if len(filtered_cluster_ids) > 0:
            print('The following clusters were filtered due to percentile cutoff: {}'
                  .format(filtered_cluster_ids))
        actual_best_k = len(set([cl for bbox_id, cl in id_to_cluster_label.items() if cl >= 0]))
        return new_cluster_labels, best_bbox_ids, actual_best_k


class GroupingEvaluator(object):
    def __init__(self, hyper_params):
        self.HYPER_PARAMS = hyper_params

    @staticmethod
    def unsupervised_evaluate_clusters(X, labels_pred, cluster_analysis):
        # Number of clusters in labels, ignoring noise if present.
        valid_labels_indices = [index for index, lab in enumerate(labels_pred) if lab >= 0]
        valid_clusters_predictions = [labels_pred[valid_index] for valid_index in valid_labels_indices]
        n_clusters = len(set(valid_clusters_predictions))
        n_noise = len(labels_pred) - len(valid_clusters_predictions)
        p_noise = 100.0*n_noise / X.shape[0]
        if n_clusters <= 0:
            print(Exception('All data points are noise!'))
            return None, p_noise

        # degenerate case
        if n_clusters == 1:
            print('A single cluster yields a degenerate cluster evaluation.')
            return None, p_noise

        x_without_noise = X[valid_labels_indices]
        labels_pred_without_noise = valid_clusters_predictions

        silhouette = em.silhouette_score(x_without_noise, labels_pred_without_noise)

        print('Analysing model: "{}": #clusters={:02}, #noise={}, %noise={:1.3f}, Silhouette={:10.9f}'
              .format(cluster_analysis, n_clusters, n_noise, p_noise, silhouette))
        return silhouette, p_noise

    def evaluate_clusters(self, X, labels_pred, y, video_name):
        # Number of clusters in labels, ignoring noise if present.
        valid_labeles_indices = [index for index, lab in enumerate(labels_pred) if lab >= 0]
        valid_clusters_predictions = [labels_pred[valid_index] for valid_index in valid_labeles_indices]
        n_clusters = len(set(valid_clusters_predictions))
        n_noise = len(labels_pred) - len(valid_clusters_predictions)
        if n_clusters <= 0:
            print('****** All data points are noise! ******')
            return dict(hyper_params=self.HYPER_PARAMS)

        p_noise = 100.0*n_noise / X.shape[0] if X.shape[0] > 0 else 1.

        # degenerate case
        character_set = set(y)
        n_characters = len(character_set)
        if n_clusters == 1 and n_characters == 1:
            print('A single cluster yields a degenerate cluster evaluation.')
            return dict(video_name=video_name, hyper_params=self.HYPER_PARAMS, k=n_clusters,
                        k_error=0, noise_count=n_noise, p_noise=p_noise, homogenity=1.0,
                        completeness=1.0, v_measure=1.0, random_index=1.0, adjusted_mutual_info=1.0,
                        normalized_mutual_info=1.0, silhouette=1.0, purity=1.0, itzi_k=1.0, acc=1.0, f_score=1.)

        y_without_noise = y[valid_labeles_indices]
        x_without_noise = X[valid_labeles_indices]
        labels_pred_without_noise = valid_clusters_predictions

        homogeneity = metrics.homogeneity_score(y_without_noise, labels_pred_without_noise)
        completeness = metrics.completeness_score(y_without_noise, labels_pred_without_noise)
        v_measure = metrics.v_measure_score(y_without_noise, labels_pred_without_noise)
        random_index = metrics.adjusted_rand_score(y_without_noise, labels_pred_without_noise)
        adjusted_mutual_info = metrics.adjusted_mutual_info_score(y_without_noise, labels_pred_without_noise)
        normalized_mutual_info = sklearn.metrics.normalized_mutual_info_score(y_without_noise, labels_pred_without_noise)
        silhouette = em.silhouette_score(x_without_noise, labels_pred_without_noise)
        itzi_k = em.itzi_k(y_without_noise, labels_pred_without_noise)
        k_error = len(set(labels_pred_without_noise)) - len(set(y_without_noise))
        cluster_purity = em.purity(y_without_noise, labels_pred_without_noise)
        acc=em.clustering_accuracy(y_without_noise, labels_pred_without_noise)
        f_score=em.f_score(y_without_noise, labels_pred_without_noise)
        eval_metrics = dict(video_name=video_name, hyper_params=self.HYPER_PARAMS, k=n_clusters,
                            k_error=k_error, noise_count=n_noise, p_noise=p_noise, homogenity=homogeneity,
                            completeness=completeness, v_measure=v_measure, random_index=random_index,
                            adjusted_mutual_info=adjusted_mutual_info, normalized_mutual_info=normalized_mutual_info,
                            silhouette=silhouette, purity=cluster_purity, itzi_k=itzi_k, acc=acc, f_score=f_score)

        print('Analysing model: "{}": k_error={:02}, #clusters={:02}, #noise={}, %noise={:1.3f} Homogenity={:10.9f},'
              ' Completeness={:10.9f}, V={:10.9f}, RI={:10.9f}, AMI={:10.9f}, NMI={:10.9f}, Silhouette={:10.9f},'
              ' Purity={:10.9f}, Itzi_k={:10.9f}, acc={:10.9f}, f_score={:10.9f}'
              .format(self.HYPER_PARAMS['cluster_analysis'], k_error, n_clusters, n_noise, p_noise, homogeneity,
                      completeness, v_measure, random_index, adjusted_mutual_info, normalized_mutual_info, silhouette,
                      cluster_purity, itzi_k, acc, f_score))
        return eval_metrics
