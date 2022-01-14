import os
import shutil
import time
from collections import OrderedDict

from Tracker.shot_associator import ShotAssociator
from Tracker.tracks_data import ShotMotAssignment, FrameDetections
from Animator.bbox_grouper_api import CharacterDetectionOutput
from Animator.utils import serialize_pickle, deserialize_pickle


class SingleShotTracker:
    def __init__(self, detection_dir_name: str, frames_dir_name: str, shot_to_mot_pkl_name: str, sampled_fps: float,
                 shot_detections_fn: str, vid_root: str, shots_repo: str):
        """
        run MOT on a single shot
        :param detection_dir_name: 'det'
        :param frames_dir_name: 'img1'
        :param shot_to_mot_pkl_name: the mot pickle path of this shot
        :param sampled_fps: the sampled frame-rate
        :param shot_detections_fn: the path to the detection json
        :param vid_root: the root repo path
        :param shots_repo: the shots repo path
        """
        self.frame_to_detections = ShotMotAssignment()
        self.detection_dir_name = detection_dir_name
        self.frames_dir_name = frames_dir_name
        self.shot_to_mot_pkl_name = shot_to_mot_pkl_name
        self.sampled_fps = sampled_fps
        self.shot_detections_fn = shot_detections_fn
        self.vid_root = vid_root
        self.shots_repo = shots_repo
        self.frames_to_detections = None
        self.box_to_track = OrderedDict()
        self.offline_mot = None

    def track_single_shot(self):
        """
        run offline MOT on a single shot
        :return: None. It updates the properties: shot_to_frames_to_detections, shot_to_offline_mot, and
            shot_to_box_to_track
        """
        start_time = time.time()

        # shot level association files
        shot, shot_mot_pkl_output, frames_to_detections_pkl, box_to_track_pkl, offline_mot_pkl = \
            self._get_pickles()
        shot_dir = os.path.join(self.vid_root, self.shots_repo, shot)
        shot_frames_dir = os.path.join(shot_dir, self.frames_dir_name)

        # in case the shot's MOT have already been calculated just load it otherwise find it
        if self.try_loading_single_shot(frames_to_detections_pkl, box_to_track_pkl, offline_mot_pkl):
            return shot, self.offline_mot, self.frames_to_detections, self.box_to_track

        # pass once to populate the frame paths
        for frame_name in os.listdir(shot_frames_dir):
            frame_path = os.path.join(shot_frames_dir, frame_name)
            self.frame_to_detections.frame_mot[frame_name] = FrameDetections(frame_path)

        # second pass to populate the detections
        try:
            detection_output = CharacterDetectionOutput.read_from_json(self.shot_detections_fn)
        except Exception as e:
            mal_det_repo, _ = os.path.split(self.shot_detections_fn)
            shutil.rmtree(mal_det_repo)
            os.mkdir(mal_det_repo)
            print(f"failed deserializing detection json: {self.shot_detections_fn}. Removed {mal_det_repo}."
                  f" Re-run main now...")
            raise e
        for bbox in detection_output.CharacterBoundingBoxes:
            self.frame_to_detections.frame_mot[bbox.KeyframeThumbnailId].detections.append(bbox)

        self.offline_mot = self._shot_offline_mot(detection_output, shot_mot_pkl_output)

        track_id = 0
        for track in self.offline_mot:
            for box_id in track:
                self.box_to_track[box_id] = track_id
            track_id += 1

        # serialize partial solution for future run
        serialize_pickle(frames_to_detections_pkl, self.frame_to_detections)
        serialize_pickle(box_to_track_pkl, self.box_to_track)
        serialize_pickle(offline_mot_pkl, self.offline_mot)
        print(f'optimizing shot {shot} took {time.time()-start_time:.3f} sec.')
        return shot, self.offline_mot, self.frame_to_detections, self.box_to_track

    def _get_pickles(self):
        """returns the paths of the temporary pickle files of this shot's MOT"""
        shot = self.shot_detections_fn.split(self.shots_repo)[-1].split(self.detection_dir_name)[0].replace('\\', '')
        shot_dir = os.path.join(self.vid_root, self.shots_repo, shot)
        shot_mot_pkl_output = os.path.join(shot_dir, self.shot_to_mot_pkl_name.format(shot))

        frames_to_detections_pkl = os.path.join(shot_dir, 'shot_to_frames_to_detections.pkl')
        box_to_track_pkl = os.path.join(shot_dir, 'shot_to_box_to_track.pkl')
        offline_mot_pkl = os.path.join(shot_dir, 'shot_to_offline_mot.pkl')
        return shot, shot_mot_pkl_output, frames_to_detections_pkl, box_to_track_pkl, offline_mot_pkl

    def is_missing_pickles(self):
        """verifier of the 4 pickle files in the shot's repo"""
        _, shot_mot_pkl_output, frames_to_detections_pkl, box_to_track_pkl, offline_mot_pkl = \
            self._get_pickles()
        pickles = [shot_mot_pkl_output, frames_to_detections_pkl, box_to_track_pkl, offline_mot_pkl]
        for pkl in pickles:
            if not os.path.isfile(pkl):
                return True
        return False

    def try_loading_single_shot(self, shot_to_frames_to_detections_pkl, shot_to_box_to_track_pkl,
                                shot_to_offline_mot_pkl):
        """load shot association data if exist"""
        pkl_files = [shot_to_frames_to_detections_pkl, shot_to_box_to_track_pkl, shot_to_offline_mot_pkl]
        if all([os.path.isfile(p) for p in pkl_files]):
            self.frames_to_detections = deserialize_pickle(shot_to_frames_to_detections_pkl)
            self.box_to_track = deserialize_pickle(shot_to_box_to_track_pkl)
            self.offline_mot = deserialize_pickle(shot_to_offline_mot_pkl)
            return True
        return False

    def _shot_offline_mot(self, detections, shot_mot_pickle_path):
        """run a second pass over the tracks for smoothing and validation"""
        # find noisy tracks and filter
        associator = ShotAssociator(detections, self.sampled_fps, shot_mot_pickle_path)
        shot_offline_tracks = associator.associate_shot_boxes()
        return shot_offline_tracks
