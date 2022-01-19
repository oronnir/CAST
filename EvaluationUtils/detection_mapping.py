import json
import os
from Animator.utils import eprint
import traceback


class DetectionMapping:
    def __init__(self, bbox_id, key_frame_thumbnail_id, keyframe_width, keyframe_height, x, y, width, height,
                 detection_confidence, thumbnail_id, bbox_group=None, is_best=None):
        self.Id = bbox_id
        self.KeyFrameThumbnailId = key_frame_thumbnail_id
        self.KeyFrameWidth = keyframe_width
        self.KeyFrameHeight = keyframe_height
        self.X = x
        self.Y = y
        self.Width = width
        self.Height = height
        self.DetectionConfidence = detection_confidence
        self.ThumbnailId = thumbnail_id
        self.BboxGroup = bbox_group
        self.IsBest = is_best

    @staticmethod
    def read_detection_input_json(json_path):
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r") as text_file:
                    json_dict = json.load(text_file)
                return json_dict['KeyFrameWidth'], json_dict['KeyFrameHeight'], json_dict['KeyFrames']
            except Exception as e:
                traceback.print_exc()
                eprint(' with exception: \'{}\'' % e)
        return None, None, None

    @staticmethod
    def read_detection_output_json(json_path, groups):
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r") as text_file:
                    detection_json_dict = json.load(text_file)
                kf_width = detection_json_dict['nativeKeyframeWidth']
                kf_height = detection_json_dict['nativeKeyframeHeight']
                mapping = []
                thumbnailid_to_detection = {detection['thumbnailId']: detection
                                            for detection in detection_json_dict['characterBoundingBoxes']}
                for bbox_thumbnailid in [bbox['thumbnailId'] for bbox in detection_json_dict['characterBoundingBoxes']]:
                    bbox = thumbnailid_to_detection[bbox_thumbnailid]
                    kf_thumbnail_id = bbox['keyframeThumbnailId']
                    if bbox_thumbnailid not in groups:
                        continue
                    bbox_group_info = groups[bbox_thumbnailid]
                    mapping.append(DetectionMapping(bbox['id'], kf_thumbnail_id, kf_width, kf_height, bbox['rect']['x'],
                                                    bbox['rect']['y'], bbox['rect']['width'], bbox['rect']['height'],
                                                    bbox['confidence'], bbox['thumbnailId'],
                                                    bbox_group_info['ClusterId'], bbox_group_info['IsBest']))

                return mapping
            except Exception as e:
                traceback.print_exc()
                eprint(' with exception: \'{}\'' % e)
        return None

    @staticmethod
    def read_consolidation_output_json(json_path):
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r") as text_file:
                    consolidation_json_dict = json.load(text_file)

                # make bbox Ids and names zero-based
                zero_based_consolidations = []
                for bbox in consolidation_json_dict['BoundingBoxesGroups']:
                    bbox['Id'] -= 1
                    bbox['BboxName'] -= 1
                    zero_based_consolidations.append(bbox)

                bbox_id_to_group = {bbox['ThumbnailId']:  bbox for bbox in zero_based_consolidations}
                return bbox_id_to_group
            except Exception as e:
                traceback.print_exc()
                eprint(' with exception: \'{}\'' % e)
        return None

    @staticmethod
    def parse_index(detection_output, grouping_output):
        if not os.path.isfile(detection_output) or \
                not os.path.isfile(grouping_output):
            return None

        groups_mappings = DetectionMapping.read_consolidation_output_json(grouping_output)
        detection_mappings = DetectionMapping.read_detection_output_json(detection_output, groups_mappings)
        return detection_mappings

    @staticmethod
    def parse_negatives(grouping_output_path):
        if os.path.isfile(grouping_output_path):
            try:
                with open(grouping_output_path, "r") as text_file:
                    json_dict = json.load(text_file)
                negative_examples = json_dict['BackgroundNegativeExamples']
                return negative_examples
            except Exception as e:
                traceback.print_exc()
                eprint(' with exception: \'{}\'' % e)
        return None
