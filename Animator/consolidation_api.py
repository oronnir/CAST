import os
import json
import copy
import numpy as np

import EvaluationUtils.vision_metrics
from Animator.utils import eprint, convert_to_dict, to_json
from argparse import _AttributeHolder


# Input objects
class BoundingBox(_AttributeHolder):
    def __init__(self, x, y, w, h):
        self.X = x
        self.Y = y
        self.Width = w
        self.Height = h

    def area(self) -> int:
        return self.Width * self.Height

    def center(self) -> tuple:
        return self.X+self.Width/2, self.Y+self.Height/2

    def center_distance(self, other):
        this_center = np.asarray(self.center())
        other_center = np.asarray(other.center())
        return np.linalg.norm(this_center - other_center)

    def to_p2_format(self):
        """return [x1, y1, x2, y2]"""
        return self.X, self.Y, self.X+self.Width, self.Y+self.Height

    def iou(self, other) -> float:
        """intersection over union of axis aligned bounding boxes"""
        iou = EvaluationUtils.vision_metrics.CVMetrics.bb_intersection_over_union(self, other)
        return iou


class CharacterBoundingBox(_AttributeHolder):
    """
    Per bounding box input data. In case the field Character is available, the input is considered labeled for
    clustering evaluation.
    """

    def __init__(self, json_dict):
        self.Id = json_dict['id']
        self.IdInFrame = -1
        self.TrackId = -1
        if 'file' in json_dict:
            self.File = json_dict['file']
        self.KeyFrameIndex = json_dict['keyFrameIndex']
        self.KeyframeThumbnailId = json_dict['keyframeThumbnailId'] if 'keyframeThumbnailId' in json_dict else None
        self.Rect = BoundingBox(json_dict['x'], json_dict['y'], json_dict['width'], json_dict['height']) \
            if 'x' in json_dict \
            else BoundingBox(json_dict['rect']['x'], json_dict['rect']['y'], json_dict['rect']['width'],
                             json_dict['rect']['height'])
        self.Confidence = json_dict['confidence']
        self.Features = np.nan_to_num(np.asarray(json_dict['features']))
        self.ThumbnailId = json_dict['thumbnailId'] if 'thumbnailId' in json_dict else None

        # label - available only for training!
        self.IsLabeled = 'character' in json_dict
        if self.IsLabeled:
            self.Character = json_dict['character']
            self.IoU = json_dict.get('iou', json_dict['ioU'])
            self.X_tsv = json_dict.get('x_tsv', json_dict['xTsv'])
            self.Y_tsv = json_dict.get('y_tsv', json_dict['yTsv'])
            self.Width_tsv = json_dict.get('width_tsv', json_dict['widthTsv'])
            self.Height_tsv = json_dict.get('height_tsv', json_dict['heightTsv'])


class CharacterDetectionOutput(_AttributeHolder):
    """
    The input object for the grouping part
    """

    def __init__(self, character_bounding_boxes):
        self.CharacterBoundingBoxes = list()

        # update keyframe index counters
        kf_counter = 0
        kf_id_to_index = dict()
        box_in_frame = 0

        # chronological order
        sorted_boxes = [CharacterBoundingBox(box) for box in sorted(character_bounding_boxes['characterBoundingBoxes'],
                                                                    key=lambda x: [x['keyframeThumbnailId'], x['id']])]
        for box in sorted_boxes:
            # update keyframe index
            if box.KeyframeThumbnailId in kf_id_to_index:
                box.KeyFrameIndex = kf_id_to_index[box.KeyframeThumbnailId]
                box_in_frame += 1
            else:
                box_in_frame = 0
                kf_id_to_index[box.KeyframeThumbnailId] = kf_counter
                kf_counter += 1

            box.KeyFrameIndex = kf_id_to_index[box.KeyframeThumbnailId]
            box.IdInFrame = box_in_frame
            self.CharacterBoundingBoxes.append(box)

        # keyframe native size
        self.NativeKeyframeWidth = character_bounding_boxes\
            .get('keyFrameWidth', character_bounding_boxes.get('nativeKeyframeWidth', 320))
        self.NativeKeyframeHeight = character_bounding_boxes\
            .get('keyFrameHeight', character_bounding_boxes.get('nativeKeyframeHeight', 180))

    @classmethod
    def read_from_json(cls, json_path):
        if not os.path.isfile(json_path):
            return None
        try:
            with open(json_path, "r") as text_file:
                json_dict = json.load(text_file)
            character_detections = cls(json_dict)
            character_detections.CharacterBoundingBoxes = sorted(character_detections.CharacterBoundingBoxes,
                                                                 key=lambda box: (box.KeyFrameIndex, box.IdInFrame))
            return character_detections
        except Exception as e:
            eprint(f'CharacterDetectionOutput.read_from_json("{json_path}") failed with exception:{os.linesep}', e)
            return None

    def save_as_json(self, json_path):
        """save as json"""
        self_clone = copy.deepcopy(self)
        for cbb in self_clone.CharacterBoundingBoxes:
            cbb.Features = [float(f) for f in cbb.Features]
        to_json(self_clone, json_path)
        return

    def consolidate_by_keyframes(self):
        """
        index the detections w.r.t. their keyframes
        :return: the dictionary index
        """
        from itertools import groupby
        keyframe_to_bboxes = dict()
        for key, group in groupby(self.CharacterBoundingBoxes, lambda cbb: cbb.KeyframeThumbnailId):
            keyframe_to_bboxes[key] = list(group)
        return keyframe_to_bboxes


# Output objects
class CharacterConsolidationOutput(_AttributeHolder):
    """
    The clustering output object
    """

    def __init__(self, bounding_boxes, background_negative_examples=None):
        self.BoundingBoxes = bounding_boxes
        self.BackgroundNegativeExamples = background_negative_examples

    def serialize(self):
        return convert_to_dict(self)


class BackgroundNegativeExample(_AttributeHolder):
    def __init__(self, keyframe_id, x, y, width, height):
        self.KeyframeId = keyframe_id
        self.BoundingBox = BoundingBox(x, y, width, height)

    def to_dict(self):
        return dict(KeyframeId=self.KeyframeId, BoundingBox=self.BoundingBox.__dict__)


class ConsolidationBoundingBox(_AttributeHolder):
    def __init__(self, bbox_id, cluster_id, is_best):
        self.Id = bbox_id
        self.ClusterId = cluster_id
        self.IsBest = is_best
