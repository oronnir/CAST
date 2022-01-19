import numpy as np
import sklearn.metrics
from numpy import linalg as LA
from sklearn import metrics

import Animator.evaluation_metrics as em

RANDOM_SEED = 1234567
np.random.seed(RANDOM_SEED)


class CharacterConsolidator(object):
    def __init__(self, detected_bboxes, cluster_analysis, normalization_method, min_cluster_significance,
                 keep_cluster_percentile, min_cluster_size):
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


class ConsolidationEvaluator(object):
    def __init__(self, hyper_params):
        self.HYPER_PARAMS = hyper_params

    @staticmethod
    def unsupervised_evaluate_clusters(x, labels_pred, cluster_analysis):
        # Number of clusters in labels, ignoring noise if present.
        valid_labels_indices = [index for index, lab in enumerate(labels_pred) if lab >= 0]
        valid_clusters_predictions = [labels_pred[valid_index] for valid_index in valid_labels_indices]
        n_clusters = len(set(valid_clusters_predictions))
        n_noise = len(labels_pred) - len(valid_clusters_predictions)
        p_noise = 100.0 * n_noise / x.shape[0]
        if n_clusters <= 0:
            print(Exception('All data points are noise!'))
            return None, p_noise

        # degenerate case
        if n_clusters == 1:
            print('A single cluster yields a degenerate cluster evaluation.')
            return None, p_noise

        x_without_noise = x[valid_labels_indices]
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
