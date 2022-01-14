import math
import numpy as np
from random import randint, seed
from copy import deepcopy
from typing import List

from EvaluationUtils.vision_metrics import CVMetrics
from Animator.bbox_grouper_api import CharacterBoundingBox
from Animator.utils import serialize_pickle, deserialize_pickle

seed(1234567)


class Triplet:
    def __init__(self, shot_id, pos, anc, neg):
        self.shot_id = shot_id
        self.positive = pos
        self.anchor = anc
        self.negative = neg

    def __repr__(self):
        return f'Triplet({self.shot_id}, Pos: {self.positive}, Anchor: {self.anchor}, Neg: {self.negative})'


class FrameTrack:
    """
    mapping the frame-level information
    """
    def __init__(self, video_level_index: int, frame_detections: List[CharacterBoundingBox]):
        self.video_level_index = video_level_index
        self.frame_detections = frame_detections


class ShotTrack:
    """
    mapping the shot-level information
    """
    def __init__(self, shot_name: str, frame_tracks: List[FrameTrack]):
        self.shot_name = shot_name
        self.frame_tracks = frame_tracks


class VideoTrack:
    """
    mapping the video-level information
    """
    def __init__(self, video_path: str, fps: float, frame_width: int, frame_height: int, shot_tracks: List[ShotTrack]):
        self.video_path = video_path
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.shot_tracks = shot_tracks

    def serialize(self, output_path: str) -> None:
        serialize_pickle(self, output_path)

    def filter_for_sampling(self, max_gap_per_sec: float, fade_frames: int):
        """filter untracked detections, possibly swapping tracks, and shots of a single track"""
        shot_to_tracks = dict()
        shot_to_multi_track_frames = dict()
        for shot_track in self.shot_tracks:
            shot_track_ids = set()
            track_to_max_jump = dict()
            track_to_detections = dict()
            multi_track_frames = dict()

            # fade-in/fade-out stats
            start_frame = shot_track.frame_tracks[0].video_level_index
            end_frame = shot_track.frame_tracks[-1].video_level_index

            # ignore too short shots
            if end_frame - start_frame < self.fps:
                continue

            # remove track-less detections
            for original_frame_track in shot_track.frame_tracks:
                # avoid fade-in/out frames
                if start_frame + fade_frames > original_frame_track.video_level_index > end_frame - fade_frames:
                    continue
                filtered_frame_track = []
                for box in original_frame_track.frame_detections:
                    if not hasattr(box, 'TrackId') or box.TrackId < 0:
                        continue
                    rect_scale = math.sqrt(box.Rect.area())
                    current_gap = track_to_max_jump.get(box.TrackId,
                                                        {'frame_id': original_frame_track.video_level_index,
                                                         'gap': 0, 'rectangle': box.Rect, 'scale': rect_scale,
                                                         'gap_ratio': 0})
                    if current_gap['frame_id'] != original_frame_track.video_level_index:
                        gap = box.Rect.center_distance(current_gap['rectangle'])
                        frame_skip = original_frame_track.video_level_index - current_gap['frame_id']
                        gap_ratio = gap/rect_scale/frame_skip
                        if gap_ratio > current_gap['gap_ratio']:
                            current_gap['gap'] = gap
                            current_gap['gap_ratio'] = gap_ratio
                            current_gap['rectangle'] = deepcopy(box.Rect)
                            current_gap['frame_id'] = original_frame_track.video_level_index
                            current_gap['scale'] = rect_scale

                    track_to_max_jump[box.TrackId] = current_gap
                    filtered_frame_track.append(box)
                    shot_track_ids.add(box.TrackId)
                original_frame_track.frame_detections = filtered_frame_track

            # remove noisy tracks (with possible id swaps)
            tracks_to_discard = set()
            for track_id, gap_stats in track_to_max_jump.items():
                if gap_stats['gap_ratio']*self.fps > max_gap_per_sec:
                    tracks_to_discard.add(track_id)
            for original_frame_track in shot_track.frame_tracks:
                filtered_frame_track = []
                for box in sorted(original_frame_track.frame_detections, key=lambda b: b.Rect.area(), reverse=True):
                    # keep up to 50% IoU boxes
                    if box.TrackId in tracks_to_discard or \
                            len([b for b in filtered_frame_track
                                 if CVMetrics.bb_intersection_over_union(b.Rect, box.Rect) > .4]) > 0:
                        continue
                    filtered_frame_track.append(box)
                    track_to_detections[box.TrackId] = track_to_detections.get(box.TrackId, []) + [box]
                if len(filtered_frame_track) >= 2:
                    multi_track_frames[original_frame_track.video_level_index] = \
                        [{'TrackId': b.TrackId, 'ThumbnailId': b.ThumbnailId,
                          'KeyframeThumbnailId': b.KeyframeThumbnailId}
                         for b in filtered_frame_track]
                original_frame_track.frame_detections = filtered_frame_track

            shot_to_multi_track_frames[shot_track.shot_name] = multi_track_frames

            # remove tracks with less than 2 tracks (for negative)
            if len(shot_track_ids - {-1}) < 2:
                shot_track.frame_tracks = []
                track_to_detections = None

            # index from shot to TrackId to detections for sampling
            shot_to_tracks[shot_track.shot_name] = track_to_detections
        return shot_to_tracks, shot_to_multi_track_frames

    def sample_triplets(self, n_triplets: int, max_gap_per_sec: float, fade_frames: int) -> List[Triplet]:
        """
        Sample n shot-level triplets
        :param fade_frames: number of frames to not sample from on intro and fade out of each shot
        :param max_gap_per_sec: The maximal normalized center distance to be considered as a non-swapped track.
        The normalization is per the scale of the bounding box and the FPS.
        :param n_triplets: The total triplets to sample
        :return: shot to triplets
        """
        shots_to_tracks_to_detections, multi_track_shots = self.filter_for_sampling(max_gap_per_sec, fade_frames)
        print('Start sampling triplets...')
        triplets = []
        indexed_shots_to_tracks_to_detections = [s for s in shots_to_tracks_to_detections
                                                 if shots_to_tracks_to_detections[s] is not None and
                                                 len(shots_to_tracks_to_detections) > 1 and
                                                 len(multi_track_shots[s]) > 0]
        m = len(indexed_shots_to_tracks_to_detections)
        for i in range(n_triplets):
            # sample a shot
            shot_name = indexed_shots_to_tracks_to_detections[i % m]
            shot_tracks = shots_to_tracks_to_detections[shot_name]

            # sample anchor and a negative examples from the same frame
            anc_neg_frame = multi_track_shots[shot_name]
            multi_track_frame_index = randint(0, len(anc_neg_frame)-1)
            candidates_detections = list(anc_neg_frame.values())[multi_track_frame_index]
            anchor, negative = np.random.choice(candidates_detections, 2, replace=False)

            # pick the positive example
            anc_track = shot_tracks[anchor['TrackId']]
            positive = anc_track[randint(0, len(anc_track)-1)]

            # assign
            triplets.append(Triplet(shot_name, positive.ThumbnailId, anchor['ThumbnailId'], negative['ThumbnailId']))

        return triplets

    @staticmethod
    def deserialize(pkl_path: str):
        return deserialize_pickle(pkl_path)

    def __repr__(self):
        return f'VideoTrack({self.video_path})'
