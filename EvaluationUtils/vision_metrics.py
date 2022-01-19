import numpy as np
from numpy import dot
from numpy.linalg import norm


class CVMetrics:
    @staticmethod
    def cosine_similarity(a, b):
        return dot(a, b)/(norm(a)*norm(b))

    @staticmethod
    def range_intersection(a_start, a_end, b_start, b_end):
        if a_end < b_start or a_start > b_end:
            return 0

        last = min(a_end, b_end)
        first = max(a_start, b_start)
        return last - first

    @staticmethod
    def bb_intersection_over_union(box_a, box_b):
        """intersection over union of axis aligned bounding boxes"""
        horizontal_intersection = CVMetrics.range_intersection(box_a.X, box_a.X + box_a.Width, box_b.X,
                                                               box_b.X + box_b.Width)
        vertical_intersection = CVMetrics.range_intersection(box_a.Y, box_a.Y + box_a.Height, box_b.Y,
                                                             box_b.Y + box_b.Height)

        intersection_area = horizontal_intersection * vertical_intersection
        union_area = box_a.area() + box_b.area() - intersection_area
        iou = float(intersection_area) / union_area
        return iou

    @staticmethod
    def matching_bbox_sets(bbox_arr_a, bbox_arr_b, min_iou):
        ious = np.zeros([len(bbox_arr_a), len(bbox_arr_b)])
        for index_a in range(len(bbox_arr_a)):
            for index_b in range(len(bbox_arr_b)):
                ious[index_a, index_b] = CVMetrics.bb_intersection_over_union(bbox_arr_a[index_a], bbox_arr_b[index_b])
        ious_copy = ious.copy()
        a_to_b_match = dict()
        while ious.any():
            linear_index = ious.argmax()
            x, y = np.unravel_index(linear_index, shape=[len(bbox_arr_a), len(bbox_arr_b)])
            if ious[x, y] < min_iou:
                return a_to_b_match, ious_copy

            a_to_b_match[x] = y
            ious[x, :] = 0
            ious[:, y] = 0

        return a_to_b_match, ious_copy

    @staticmethod
    def precision_recall_at_iou(gt_dict, pred_dict, min_iou):
        ids_superset = set(gt_dict.keys()) | set(gt_dict.keys())
        ordered_ids = sorted(ids_superset)[5:-5]

        # book level/video level
        fp = 0
        tp = 0
        fn = 0
        for frame_id in ordered_ids:
            if frame_id not in gt_dict:
                fp += len(pred_dict)
                continue
            if frame_id not in pred_dict:
                fn += len(gt_dict)
                continue
            ordered_gt = list(gt_dict[frame_id].values())
            ordered_pred = list(pred_dict[frame_id].values())
            gt_pred_match, _ = CVMetrics.matching_bbox_sets(ordered_gt, ordered_pred, min_iou)
            tp += len(gt_pred_match)
            missed_boxes = len(ordered_gt) - len(gt_pred_match)
            false_detections = len(ordered_pred) - len(gt_pred_match)
            if missed_boxes < 0 or false_detections < 0:
                stop = 1
            fp += false_detections
            fn += missed_boxes

        precision = 1.0 * tp / (fp + tp)
        recall = 1.0 * tp / (tp + fn)
        return precision, recall
