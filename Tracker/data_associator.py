import glob
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Tuple
from Tracker.demonstrator import DemoCompiler
from Tracker.single_shot_tracker import SingleShotTracker
from Tracker.tracks import VideoTrack, ShotTrack, FrameTrack
from Tracker.tracks_data import VideoMotAssignment
from Utils import multiprocess_worker as mp
from Detector.detector_wrapper import DETECTION_JSON_NAME
from Animator.utils import sort_ordered_dict_by_key, serialize_pickle


class MultiShotMot:
    """
    video level MOT
    """
    FRAMES_DIR_NAME = 'img1'
    DETECTION_DIR_NAME = 'det'
    VISUALIZATION_DIR_NAME = 'visualization'
    INTERNAL_OUTPUT_PICKLE = 'internal_output.pkl'
    EXTERNAL_VIDEO_TRACK_OUTPUT_PICKLE = 'video_track_external_{}.pkl'
    OUTPUT_DIR_NAME = 'output'
    SINGLE_SHOT_TO_MOT_PKL = 'shot_pkl_{}.pkl'

    def __init__(self, video_path: str, sampled_fps: float, out_fps: float,  width: int, height: int, output_folder: str):
        """
        Offline Multi-Object Tracker for multiple shots
        :param video_path:
        :param sampled_fps: the sampling rate of frames
        :param out_fps: the output fps for the optional demo video if should_generate_video
        :param width:
        :param height:
        :param output_folder:
        """
        self.video_path = video_path
        self.sampled_fps = sampled_fps
        self.out_fps = out_fps
        self.width = width
        self.height = height
        self.output_folder = output_folder
        self.mot = None
        self.shot_to_frames_to_detections = VideoMotAssignment()
        self.shot_to_box_to_track = OrderedDict()
        self.shot_to_offline_mot = OrderedDict()

    def track_multi_object_per_shot(self, vid_root: str, shots_repo: str, should_generate_video: bool) -> Tuple[VideoTrack, str]:
        """
        run offline MOT per shot and generate visualization if needed
        :param vid_root: the top level directory
        :param shots_repo: the shots repo name
        :param should_generate_video: Boolean
        :return: a shot_to_box_to_track mapping
        """
        # process tracking in parallel
        pattern = os.path.join(vid_root, shots_repo, '*', MultiShotMot.DETECTION_DIR_NAME, DETECTION_JSON_NAME)
        detection_paths = sorted(glob.glob(pattern))
        tracks_kwargs = [{'shot_tracker': SingleShotTracker(MultiShotMot.DETECTION_DIR_NAME,
                                                            MultiShotMot.FRAMES_DIR_NAME,
                                                            MultiShotMot.SINGLE_SHOT_TO_MOT_PKL, self.sampled_fps, p,
                                                            vid_root, shots_repo)}
                         for p in detection_paths]

        num_jobs = sum([tracker['shot_tracker'].is_missing_pickles() for tracker in tracks_kwargs])
        track_results = []
        if num_jobs <= 1:
            for tracker in tracks_kwargs:
                current_result = tracker['shot_tracker'].track_single_shot()
                track_results.append(current_result)
        else:
            num_processes = min(5, max(2, num_jobs))
            semaphore = mp.Semaphore(n_processes=num_processes)

            def track(shot_tracker):
                return shot_tracker.track_single_shot()

            track_results = semaphore.parallelize(tracks_kwargs, track)

        # get the outputs of track and merge
        self._internal_results_packing(track_results)

        # compile a video
        if should_generate_video:
            visualization_dir = os.path.join(vid_root, MultiShotMot.VISUALIZATION_DIR_NAME)
            demo_compiler = DemoCompiler(self.sampled_fps, self.out_fps, vid_root, self.output_folder)
            demo_compiler.generate_demo(self.shot_to_frames_to_detections, self.shot_to_box_to_track, visualization_dir)

        video_track = self._external_results_packing()

        # clear memory
        # del self.shot_to_box_to_track
        # del self.shot_to_offline_mot
        # del self.shot_to_frames_to_detections
        # import gc
        # gc.collect()
        pkl_path = self._serialize_output(video_track)

        return video_track, pkl_path

    def _sort_keys(self) -> None:
        """update the order as race conditions may change the chronological timeline of the video"""
        self.shot_to_frames_to_detections.sort_keys()
        self.shot_to_box_to_track = sort_ordered_dict_by_key(self.shot_to_box_to_track)
        self.shot_to_offline_mot = sort_ordered_dict_by_key(self.shot_to_offline_mot)

    def _internal_results_packing(self, results) -> None:
        for shot, offline_mot, frame_to_detections, box_to_track in results:
            self.shot_to_frames_to_detections.shot_mot[shot] = frame_to_detections
            self.shot_to_offline_mot[shot] = offline_mot
            self.shot_to_box_to_track[shot] = box_to_track
        self._sort_keys()

    def _external_results_packing(self) -> VideoTrack:
        """
        populate all information into the output class VideoTrack
        """
        video_level_frame_index = 0
        video_level_box_id = 0
        shot_tracks = []
        for shot, f_to_detections in self.shot_to_frames_to_detections.shot_mot.items():
            frame_tracks = []
            for frame, detections in f_to_detections.frame_mot.items():
                copied_detections = []
                for box in detections.detections:
                    box_clone = deepcopy(box)
                    if box.Id not in self.shot_to_box_to_track[shot]:
                        continue
                    box_clone.TrackId = self.shot_to_box_to_track[shot][box.Id]
                    box_clone.Id = video_level_box_id
                    box_clone.Features = None
                    copied_detections.append(box_clone)
                    video_level_box_id += 1
                frame_track = FrameTrack(video_level_frame_index, copied_detections)
                frame_tracks.append(frame_track)
                video_level_frame_index += 1
            shot_track = ShotTrack(shot, frame_tracks)
            shot_tracks.append(shot_track)
        video_track = VideoTrack(self.video_path, self.sampled_fps, self.width, self.height, shot_tracks)
        return video_track

    def _serialize_output(self, video_track: VideoTrack) -> str:
        """a video level data association serialization"""
        # serialize the external output data class VideoTrack
        internal_output_path = os.path.join(self.output_folder, MultiShotMot.INTERNAL_OUTPUT_PICKLE)
        serialize_pickle(internal_output_path, self.shot_to_frames_to_detections)
        print(f'successfully serialized the internal offline MOT output pickle into: {internal_output_path}')

        video_name = os.path.basename(self.video_path)
        external_output_path = os.path.join(self.output_folder,
                                            MultiShotMot.EXTERNAL_VIDEO_TRACK_OUTPUT_PICKLE.format(video_name))
        serialize_pickle(external_output_path, video_track)
        print(f'successfully serialized the external offline MOT output pickle into: {external_output_path}')
        return external_output_path
