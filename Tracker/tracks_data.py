from collections import OrderedDict

from Animator.utils import deserialize_pickle, sort_ordered_dict_by_key


# *** internal classes ***

class FrameDetections:
    def __init__(self, frame_path, detections=None):
        if detections is None:
            detections = []
        self.frame_path = frame_path
        self.detections = detections


class VideoMotAssignment:
    def __init__(self):
        self.shot_mot = OrderedDict()

    @staticmethod
    def unpickle(path_to_pickle):
        return deserialize_pickle(path_to_pickle)

    def sort_keys(self):
        self.shot_mot = sort_ordered_dict_by_key(self.shot_mot)


class ShotMotAssignment:
    def __init__(self):
        self.frame_mot = OrderedDict()
