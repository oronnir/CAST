import glob
import os
import tempfile
import time

from Teacher.utilizer.triplets_sampler import local_triplet_sampling
from FfmpegWrapper.ffmpeg_wrapper import FfmpegWrapper, cv2_extract_frames
from ShotWrapper import shot_extractor
from Tracker.data_associator import MultiShotMot
from Tracker.sortformatter import SortFormatter
from Detector.detector_wrapper import DetectorWrapper
from Animator.utils import create_dir_if_not_exist, eprint, profiling, colored
from Utils import multiprocess_worker as mp

frame_ext = '.jpg'
frames_folder_name = 'img1'
detections_folder_name = 'det'
detection_file_name = 'animationdetectionoutput.json'
sample_rate = 6


class VideoProcessor:
    def __init__(self):
        self.num_processes = 4

    @staticmethod
    def init_child(lock_):
        global lock
        lock = lock_

    def run_full_analysis(self, vid_path, should_visualize=False):
        """
        Run the animation pipeline Shots seg -> detection -> MOT -> Triplets sampling.
        :param vid_path: the path to the raw mp4.
        :param should_visualize: Boolean default False on whether to compile a demo vide of the MOT.
        :return: None, the output assets/artifacts are in the working folder.
        """
        try:
            time_est = time.time()
            print(f'start E2E analysis: {vid_path}')
            if not os.path.isfile(vid_path):
                raise Exception(f'File {vid_path} does not exist!')

            # stats
            ff_wrapper = FfmpegWrapper()
            vid_fps = ff_wrapper.get_frame_rate(vid_path)
            width, height = ff_wrapper.get_frame_dimensions(vid_path)
            time_est = profiling('video stats', time_est)

            # constants
            extraction_frame_rate = vid_fps / sample_rate
            if should_visualize:
                print(f'running sampling fps of {extraction_frame_rate:.3f} while the original fps is {vid_fps:.3f}.')

            # set up a working directory
            tmp = tempfile.gettempdir()
            work_dir_name = os.path.basename(vid_path).split('.')[0]
            working_dir = create_dir_if_not_exist(tmp, work_dir_name)

            # extract shots
            shots = self._extract_shots(working_dir)
            time_est = profiling('extract shots', time_est)
            print(f'found {len(shots)} shots. Start per-shot analysis...')
            video_shots = create_dir_if_not_exist(working_dir, 'video_shots')

            # extract frames
            self._extract_frames(vid_path, video_shots, shots)
            time_est = profiling('per-shot frames extraction', time_est)

            # run detection centralize
            self._detect(video_shots)
            time_est = profiling('detection and crop', time_est)

            # parallel processing per shot
            self._validate_extraction(self.num_processes, vid_path, working_dir, video_shots, extraction_frame_rate,
                                      width, height, shots)
            time_est = profiling('per-shot preprocessing', time_est)

            # offline MOT
            time_est = profiling('full tracking', time_est)
            pkl_path = self._track(working_dir, vid_path, extraction_frame_rate, vid_fps, width, height, should_visualize)
            local_triplet_sampling(pkl_path, working_dir)
            time_est = profiling('Sampling a fine-tune dataset', time_est)

            colored('***   DONE   ***', 'green')
            exit(0)
        except Exception as e:
            eprint(str(e))
            raise e

    @staticmethod
    def _track(working_dir, vid_path, extraction_frame_rate, vid_fps, width, height, should_visualize):
        shots_repo = 'video_shots'
        tracking_output_dir = create_dir_if_not_exist(working_dir, 'mot')
        multi_mot = MultiShotMot(vid_path, extraction_frame_rate, vid_fps, width, height, tracking_output_dir)
        video_track, pkl_path = multi_mot.track_multi_object_per_shot(working_dir, shots_repo, should_visualize)
        return pkl_path

    @staticmethod
    def _validate_extraction(n_processes, vid_path, working_dir, video_shots, extraction_frame_rate, width, height, shots):
        semaphore = mp.Semaphore(n_processes=n_processes)
        kv_args = [{"shot": shot, "vid_path": vid_path, "working_dir": working_dir, "video_shots": video_shots,
                    "frames_dir_name": frames_folder_name, "frame_extension": frame_ext,
                    "extraction_frame_rate": extraction_frame_rate, "width": width, "height": height}
                   for shot in shots]
        semaphore.parallelize(kv_args, VideoProcessor.analyse_single_shot, timeout=20 * 60)

    @staticmethod
    def _detect(video_shots):
        shots_folders = os.listdir(video_shots)
        input_repos = [os.path.join(video_shots, r, 'img1') for r in shots_folders]
        output_repos = [os.path.join(video_shots, r, 'det') for r in shots_folders]
        detector = DetectorWrapper()
        detector.detect_multi_repo(input_repos, output_repos)

    @staticmethod
    def _extract_shots(working_dir):
        shooter = shot_extractor.ShotExtractor()
        shots_output = create_dir_if_not_exist(working_dir, 'shots_stats')
        shots = shooter.extract_shots(video_path, shots_output)
        return shots

    @staticmethod
    def _has_detected_all(pattern, num_shots):
        """internal helper func to check if the threads have finished"""
        return len(glob.glob(pattern)) >= num_shots

    @staticmethod
    def extract_frames_single_shot(shot, vid_path, video_shots, frames_dir_name, frame_extension):
        try:
            current_shot_dir = create_dir_if_not_exist(video_shots, f'shot_{shot.id:03d}')
            frames_dir = create_dir_if_not_exist(current_shot_dir, frames_dir_name)

            # extract all frames in the shot
            num_frames = len([img for img in os.listdir(frames_dir) if img.endswith(frame_extension)])
            if num_frames == 0:
                cv2_extract_frames(vid_path, frames_dir, frame_range=(shot.frame_start, shot.frame_end))
        except Exception as e:
            eprint(str(e))
            raise e
        return

    @staticmethod
    def analyse_single_shot(shot, vid_path, working_dir, video_shots, frames_dir_name, frame_extension,
                            extraction_frame_rate, width, height):
        try:
            current_shot_dir = create_dir_if_not_exist(video_shots, f'shot_{shot.id:03d}')
            frames_dir = create_dir_if_not_exist(current_shot_dir, frames_dir_name)
            vid_name = os.path.basename(vid_path)

            # extract all frames in the shot
            num_frames = len([img for img in os.listdir(frames_dir) if img.endswith(frame_extension)])
            if num_frames == 0:
                cv2_extract_frames(vid_path, frames_dir, frame_range=(shot.frame_start, shot.frame_end))
                num_frames = len([img for img in os.listdir(frames_dir) if img.endswith(frame_extension)])

            # detect objects
            detection_dir = create_dir_if_not_exist(current_shot_dir, detections_folder_name)
            detections_path = os.path.join(detection_dir, detection_file_name)
            if not os.path.isfile(detections_path):
                detector = DetectorWrapper()
                detector.detect_and_featurize(frames_dir, detection_dir)

                # tracker files
                sort_formatter = SortFormatter()
                sort_formatter.create_ini(working_dir, vid_name, frames_dir_name, extraction_frame_rate, num_frames,
                                          width, height)
        except Exception as e:
            eprint(str(e))
            raise e
        return

    def _extract_frames(self, vid_path, video_shots, shots):
        semaphore = mp.Semaphore(n_processes=self.num_processes)
        kv_args = [{"shot": shot, "vid_path": vid_path, "video_shots": video_shots,
                    "frames_dir_name": frames_folder_name, "frame_extension": frame_ext}
                   for shot in shots]
        semaphore.parallelize(kv_args, VideoProcessor.extract_frames_single_shot, timeout=2 * 60 * 60)


if __name__ == '__main__':
    # SAIL dataset
    video_path = r"\..\videos\The Lego Movie (2014) [1080p]\TheLegoMovie_s1280.mp4"
    VideoProcessor().run_full_analysis(video_path, False)
