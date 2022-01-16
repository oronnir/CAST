import argparse
import json
import os
import random
import sys
import time
import warnings

import numpy as np

import Animator.clustering_methods as cm
import Animator.normalization_methods as nm
from Deduper.deduper import GraphAnalyzer, PranjaWrapper
from Animator.consolidation_api import CharacterDetectionOutput
from Animator.cluster_logic import CharacterGrouper
from Animator.simple_clustering import my_simple_clustering
from Animator.utils import eprint, profiling

_VERSION = '2.0.0.0'
_REQUIRED_FEATURIZER_DIM = 2048
_EDH_REPO = '???\\EDH'

# initialize seeds
SEED = 1234567


class BboxGroup(object):
    def __init__(self, bbox_id, bbox_thumbnail_id, bbox_name, cluster_id, is_best_thumbnail):
        self.Id = bbox_id
        self.ThumbnailId = bbox_thumbnail_id
        self.BboxName = bbox_name
        self.ClusterId = cluster_id
        self.IsBest = is_best_thumbnail


class BboxGrouper(object):
    def __init__(self, args, cluster_analysis, normalization_method):
        # initialize the seeds for batch purposes
        random.seed(SEED)
        np.random.seed(SEED)
        character_detections, output_path, should_yield_negative_examples, min_subframe_width, min_subframe_height, \
            min_subframe_area, max_unique_boxes, min_conf_percentile, min_short_edge_absolute, min_short_edge_ratio, \
            min_cluster_significance, max_short_edge_absolute, max_short_edge_ratio, keep_cluster_percentile, \
            min_cluster_size, cliques = read_and_validate_args(args)

        filtered_character_detections, noisy_bboxes, rep_to_dups, duplicates = BboxGrouper._filter_noisy_bboxes(
            character_detections,
            min_conf_percentile,
            min_short_edge_absolute,
            min_short_edge_ratio,
            max_short_edge_absolute,
            max_short_edge_ratio,
            cliques)

        self.ALL_CHARACTER_DETECTIONS = character_detections
        self.VALID_CHARACTER_DETECTIONS = filtered_character_detections
        self.NOISY_CHARACTER_DETECTIONS = noisy_bboxes
        self.OUTPUT_PATH = output_path
        self.GROUPER = CharacterGrouper(detected_bboxes=filtered_character_detections,
                                        cluster_analysis=cluster_analysis,
                                        normalization_method=normalization_method,
                                        min_cluster_significance=min_cluster_significance,
                                        keep_cluster_percentile=keep_cluster_percentile,
                                        min_cluster_size=min_cluster_size)
        self.SHOULD_YIELD_NEGATIVE_EXAMPLES = should_yield_negative_examples
        self.MIN_BG_WIDTH = min_subframe_width
        self.MIN_BG_HEIGHT = min_subframe_height
        self.MI_BG_AREA = min_subframe_area
        self.MAX_UNIQUE_BOXES = max_unique_boxes
        self.KEYFRAME_WIDTH = character_detections.NativeKeyframeWidth
        self.KEYFRAME_HEIGHT = character_detections.NativeKeyframeHeight
        self.DUP_CLIQUES = rep_to_dups
        self.DUPLICATES = duplicates

    @staticmethod
    def _filter_noisy_bboxes(input_detections, min_conf_percentile, min_short_edge_absolute, min_short_edge_ratio,
                             max_short_edge_absolute, max_short_edge_ratio, cliques):
        character_bbox_detections = input_detections.CharacterBoundingBoxes
        keyframe_short_edge = min(input_detections.NativeKeyframeWidth, input_detections.NativeKeyframeHeight)

        # keep only a single image per clique
        rep_to_dups = dict()
        clique_ids_to_filter = []
        for clique in cliques:
            rep_to_dups[clique[0]] = []
            for dup_thumbnail_id in clique[1:]:
                clique_ids_to_filter.append(dup_thumbnail_id)
                rep_to_dups[clique[0]].append(dup_thumbnail_id)

        dedupped_character_bbox_detections = []
        duplicates = {}
        for cd in character_bbox_detections:
            if cd.ThumbnailId in clique_ids_to_filter:
                duplicates[cd.ThumbnailId] = cd
            else:
                dedupped_character_bbox_detections.append(cd)

        # filter by detection confidence
        detection_confidences = sorted([cd.Confidence for cd in dedupped_character_bbox_detections])
        cutoff_at_max_examples = detection_confidences[0]
        max_samples = 4000
        if len(detection_confidences) > max_samples:
            cutoff_at_max_examples = detection_confidences[-max_samples]
        prior_cutoff = 0.3
        # prior_cutoff = 0.365
        min_conf_cutoff = max(prior_cutoff, cutoff_at_max_examples, np.percentile(np.asarray(detection_confidences), min_conf_percentile))
        high_confidence_character_detections = [cd for cd in dedupped_character_bbox_detections
                                                if cd.Confidence >= min_conf_cutoff]

        large_shortedge_absolute_bboxes = [cd for cd in high_confidence_character_detections
                                           if
                                           min_short_edge_absolute <= cd.Rect.Width <= max_short_edge_absolute and
                                           min_short_edge_absolute <= cd.Rect.Height <= max_short_edge_absolute]

        large_shortedge_raio_bboxes = [cd for cd in large_shortedge_absolute_bboxes
                                       if
                                       min_short_edge_ratio <= 1.*cd.Rect.Width / keyframe_short_edge <= max_short_edge_ratio and
                                       min_short_edge_ratio <= 1.*cd.Rect.Height / keyframe_short_edge <= max_short_edge_ratio]

        # keep the noisy bboxes
        valid_ids = set([vcd.Id for vcd in large_shortedge_raio_bboxes])
        noisy_bboxes = [cd for cd in character_bbox_detections if cd.Id not in valid_ids]
        num_valid_ids = len(valid_ids)
        num_noisy_ids = len(noisy_bboxes)
        print('\nOriginal number of bboxes: {}; Filtered bboxes: {}; Valid bboxes: {}'
              .format(num_valid_ids + num_noisy_ids, num_noisy_ids, num_valid_ids))
        return large_shortedge_raio_bboxes, noisy_bboxes, rep_to_dups, duplicates

    def serialize_result(self, predicted_labels_obj, k, bg_negative_examples):
        """ save resulting clusters to json """
        if os.path.isfile(self.OUTPUT_PATH):
            os.remove(self.OUTPUT_PATH)

        # serializing the characters bboxes
        bboxes_groups = []
        for bbox_group in predicted_labels_obj:
            bbox_group.BboxName = int(bbox_group.BboxName)
            bbox_group.Id = int(bbox_group.Id)
            bboxes_groups.append(bbox_group.__dict__)

        # serializing the BG negative examples
        bg_negatives = [bg_negative_example.to_dict() for bg_negative_example in bg_negative_examples]

        grouping_response = dict(NumClusters=k, BoundingBoxesGroups=bboxes_groups,
                                 BackgroundNegativeExamples=bg_negatives)
        exception_message = ''

        try:
            with open(self.OUTPUT_PATH, "w") as text_file:
                json.dump(grouping_response, text_file)

                if os.path.isfile(self.OUTPUT_PATH):
                    return 0
        except Exception as e:
            exception_message = ' with exception: \'{}\'' % e

        eprint('failed serializing output_file: \'{0}\'{1}'.format(self.OUTPUT_PATH, exception_message))
        return 1

    def find_background_negative_examples(self, min_subframe_width, min_subframe_height, min_subframe_area,
                                          max_unique_boxes):
        """
        crop all background negative examples that do not intersect with characters
        """
        all_bg_examples = []

        # This part was omitted to comply with patent's rights but has an impact only on the image classification app.

        return all_bg_examples

    def group_characters_single_video(self, time_marker):
        time_marker = profiling("Parsed args", time_marker)

        bbox_ids, predicted_labels, k, best_thumbnails = my_simple_clustering(self.GROUPER.IDS, self.GROUPER.FEATURES,
                                                                              self.GROUPER.KeyFrameIndices)
        bbox_id_to_detection = {
            detection.Id: detection for detection in self.ALL_CHARACTER_DETECTIONS.CharacterBoundingBoxes
        }

        bbox_id_to_label = dict(zip(bbox_ids, predicted_labels))
        dups_labs = dict()
        top_id = int(max(bbox_ids) + 1)
        for box_id, label in bbox_id_to_label.items():
            if label < 0:
                continue

            potential_clique = self.DUP_CLIQUES.get(bbox_id_to_detection[box_id].ThumbnailId, None)
            if potential_clique:
                for dup_thumb in potential_clique:
                    dup = self.DUPLICATES[dup_thumb]
                    dup.Id = top_id
                    top_id += 1
                    dups_labs[dup.Id] = [dup, label]

        # adding the dedupped detections with the relevant label
        bbox_ids = np.append(bbox_ids, list(dups_labs.keys()))
        for box_id, (det, lab) in dups_labs.items():
            bbox_id_to_detection[box_id] = det
        for k, v in dups_labs.items():
            bbox_id_to_label[k] = v[1]

        time_marker = profiling("group by features", time_marker)
        background_negative_examples = []
        if self.SHOULD_YIELD_NEGATIVE_EXAMPLES:
            time_marker = profiling("Calculate background negative examples", time_marker)
            background_negative_examples = self.find_background_negative_examples(self.MIN_BG_WIDTH,
                                                                                  self.MIN_BG_HEIGHT,
                                                                                  self.MI_BG_AREA,
                                                                                  self.MAX_UNIQUE_BOXES)
        bboxes_groups = []
        for bbox_id in bbox_ids:
            bbox_group = BboxGroup(bbox_id=int(bbox_id),
                                   bbox_thumbnail_id=bbox_id_to_detection[bbox_id].ThumbnailId,
                                   bbox_name=int(bbox_id),
                                   cluster_id=int(bbox_id_to_label[bbox_id]),
                                   is_best_thumbnail=bbox_id in best_thumbnails)
            bboxes_groups.append(bbox_group)

        # append all noise to output
        initial_noise_cluster_id = np.min(predicted_labels)
        for noisy_bbox_index in range(len(self.NOISY_CHARACTER_DETECTIONS)):
            bbox_group = BboxGroup(bbox_id=self.NOISY_CHARACTER_DETECTIONS[noisy_bbox_index].Id,
                                   bbox_thumbnail_id=self.NOISY_CHARACTER_DETECTIONS[noisy_bbox_index].ThumbnailId,
                                   bbox_name=self.NOISY_CHARACTER_DETECTIONS[noisy_bbox_index].Id,
                                   cluster_id=int(initial_noise_cluster_id - 1 * (noisy_bbox_index + 1)),
                                   is_best_thumbnail=True)
            bboxes_groups.append(bbox_group)
        bboxes_groups = sorted(bboxes_groups, key=lambda box: box.Id)

        if bboxes_groups is not None:
            self.serialize_result(bboxes_groups, k, background_negative_examples)
            _ = profiling("Serialized results", time_marker)
            return bboxes_groups

        _ = profiling("Failed clustering - Killing program", time_marker)
        return None


def read_and_validate_args(args):
    """
    validate cmd args for analysis of a single video
    :param args: properties
    :return: arguments or throws ValueError exception
    """
    parser = argparse.ArgumentParser(description='Bounding box grouper version: {}'.format(_VERSION))
    parser.add_argument("--input", "-i", type=str, help="Input JSON")
    parser.add_argument("--output", "-o", type=str, help="Output JSON")
    parser.add_argument("--min-confidence-percentile", type=int, default=7,
                        help="The low confidence percentile to regard as noise")

    min_absolute_short_edge = 50
    max_absolute_short_edge = 900
    parser.add_argument("--min-short-edge-absolute", type=int, default=min_absolute_short_edge,
                        help="the smallest acceptable number of pixels on edge")
    parser.add_argument("--min-short-edge-ratio", type=float, default=0.02,
                        help="the smallest acceptable ratio number of pixels on edge divided by the frame height")
    parser.add_argument("--max-short-edge-absolute", type=int, default=max_absolute_short_edge,
                        help="the largest acceptable number of pixels on the short edge")
    parser.add_argument("--max-short-edge-ratio", type=float, default=1.,
                        help="the smallest acceptable ratio number of pixels on edge divided by the frame height")
    parser.add_argument("--min-cluster-significance", type=float, default=0.5,
                        help="minimum cluster significance to not filter out")
    parser.add_argument("--keep-cluster-percentile", type=int, default=100,
                        help="The percentile of clusters to keep sorted by size descending")
    parser.add_argument("--min-cluster-size", type=int, default=3, help="minimum cluster size")

    # BG cropping args and defaults
    parser.add_argument("--neg-examples", action="store_true", help="If should yield negative examples")
    parser.add_argument("--min-width", type=int, default=90, help="Minimal subframe width")
    parser.add_argument("--min-height", type=int, default=90, help="Minimal subframe height")
    parser.add_argument("--min-area", type=int, default=10000, help="Minimal subframe area")
    parser.add_argument("--max-boxes", type=int, default=10, help="Maximal unique boxes")

    args = parser.parse_args(args[1:])

    input_json = args.input
    output_file_path = args.output
    min_conf_percentile = args.min_confidence_percentile
    min_short_edge_absolute = args.min_short_edge_absolute
    min_short_edge_ratio = args.min_short_edge_ratio
    max_short_edge_absolute = args.max_short_edge_absolute
    max_short_edge_ratio = args.max_short_edge_ratio
    min_cluster_significance = args.min_cluster_significance
    min_cluster_size = args.min_cluster_size
    keep_cluster_percentile = args.keep_cluster_percentile

    yield_negative_examples = args.neg_examples
    min_subframe_width = args.min_width
    min_subframe_height = args.min_height
    min_subframe_area = args.min_area
    max_unique_boxes = args.max_boxes

    output_file_containing_folder = os.path.abspath(os.path.join(output_file_path, os.pardir))
    if not os.path.isdir(output_file_containing_folder):
        os.mkdir(output_file_containing_folder)

    character_detections = CharacterDetectionOutput.read_from_json(input_json)

    # validate
    if character_detections is None:
        eprint("Invalid input file")
        raise ValueError('Invalid args input.')
    if len(character_detections.CharacterBoundingBoxes) == 0:
        warnings.warn('Empty input!')
    else:
        current_featurizer_dim = character_detections.CharacterBoundingBoxes[0].Features.shape[0]
        if current_featurizer_dim != _REQUIRED_FEATURIZER_DIM:
            warnings.warn('Featurizer dimension is: {} while expecting: {}'
                          .format(current_featurizer_dim, _REQUIRED_FEATURIZER_DIM))

    # EDH features based identical image exclusion
    cliques = group_identical_bboxes(character_detections, input_json)

    return character_detections, output_file_path, yield_negative_examples, min_subframe_width, min_subframe_height, \
        min_subframe_area, max_unique_boxes, min_conf_percentile, min_short_edge_absolute, min_short_edge_ratio, \
        min_cluster_significance, max_short_edge_absolute, max_short_edge_ratio, keep_cluster_percentile, \
        min_cluster_size, cliques


def group_identical_bboxes(character_detections, input_json):
    # adding EDH features - extract if not exist
    edh_json_name = "edh_features.json"
    working_folder, _ = os.path.split(input_json)
    edh_json_path = os.path.join(working_folder, edh_json_name)
    if os.path.isfile(edh_json_path):
        edh_character_detections = CharacterDetectionOutput.read_from_json(edh_json_path)
    else:
        edh_extractor = PranjaWrapper()
        edh_character_detections = edh_extractor.extract_edh_features(input_json, edh_json_path)

    id_to_edh = {bbox.ThumbnailId: bbox.Features for bbox in edh_character_detections.CharacterBoundingBoxes}
    edh_feature_matrix = np.zeros((len(edh_character_detections.CharacterBoundingBoxes),
                                   edh_character_detections.CharacterBoundingBoxes[0].Features.shape[0]))
    for i in range(len(character_detections.CharacterBoundingBoxes)):
        bbox = character_detections.CharacterBoundingBoxes[i]
        edh_features = id_to_edh[bbox.ThumbnailId]
        edh_feature_matrix[i, :] = edh_features

    graph_analyzer = GraphAnalyzer(edh_feature_matrix)
    cliques = []
    counter_col = 0
    for clique in graph_analyzer.find_cliques():
        if len(clique) > 1:
            cl_thumbnail_ids = []
            for cl_id in clique:
                cl_thumbnail_ids.append(edh_character_detections.CharacterBoundingBoxes[cl_id].ThumbnailId)
            cliques.append(cl_thumbnail_ids)
            counter_col += 1
    return cliques


def bbox_grouper_main():
    print('Start {}, version: {}'.format(__name__, _VERSION))

    # validating input
    try:
        cluster_analysis = getattr(cm, 'k_means')
        normalization_method = getattr(nm, 'identity')

        grouper = BboxGrouper(sys.argv, cluster_analysis, normalization_method)
        time_marker = time.time()
        if grouper.group_characters_single_video(time_marker) is not None:
            print('Successful run ended')
            # sys.exit(0)
        else:
            eprint('Failed running bbox_grouper')
            sys.exit(1)

    except Exception as e:
        eprint("Failed with exception {}".format(e), e)
        sys.exit(1)
