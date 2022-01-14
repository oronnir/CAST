import os
import shutil
import time
from collections import OrderedDict
from copy import deepcopy

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from FfmpegWrapper.ffmpeg_wrapper import FfmpegWrapper
from Tracker.tracks_data import VideoMotAssignment
from Utils import multiprocess_worker as mp
from Animator.utils import eprint


class DemoCompiler:
    def __init__(self, sampled_fps, out_fps, video_root, output_dir):
        """
        demo video generator
        :param sampled_fps: the frame sample rate
        :param out_fps: the output video frame rate
        :param video_root:
        :param output_dir:
        """
        self.sampled_fps = sampled_fps
        self.out_fps = out_fps
        self.num_distinct_colors = 16
        self.colors = np.random.rand(self.num_distinct_colors, 3)  # RGB
        self.frame_name_format = '%06d.jpg'
        self.visualization_video_name = 'demo.mp4'
        containing_folder, vid_name = os.path.split(video_root)
        self.original_video_name = vid_name.split(".")[0]
        demo_file_name = f'{self.original_video_name}_demo.mp4'
        self.demo_path = os.path.join(output_dir, demo_file_name)

    def generate_demo(self, shot_to_frames_to_detections: VideoMotAssignment, shot_to_offline_mot: OrderedDict,
                      working_dir: str) -> None:
        """
        compile a demo video based on the tracking output
        :param working_dir: the visualization directory in which all frames will be rendered and saved
        :param shot_to_frames_to_detections: mapping a shot to its frames and a frame to its detections
        :param shot_to_offline_mot: shot name to tracks
        :return: None (void)
        """
        start_time = time.time()
        plt.ioff()

        # clear the visualization folder
        if os.path.isdir(working_dir):
            shutil.rmtree(working_dir)
        os.mkdir(working_dir)

        video_level_frame_number = 0
        frame_index_to_kwargs = []
        for shot_name, frame_to_detections in shot_to_frames_to_detections.shot_mot.items():
            for frame_name, frame_path_and_detections in frame_to_detections.frame_mot.items():
                # making sure no resources are shared between processes!
                curr_frame_detections = deepcopy(frame_path_and_detections.detections)
                current_frame_path = frame_path_and_detections.frame_path
                current_shot_mot = OrderedDict({shot_name: deepcopy(shot_to_offline_mot[shot_name])})
                curr_frame_args = dict(frame_str=str(video_level_frame_number),
                                       frame_path=current_frame_path, shot_name=shot_name,
                                       detections=curr_frame_detections,
                                       shot_to_offline_mot=current_shot_mot, visualization_dir=working_dir)
                frame_index_to_kwargs.append(curr_frame_args)
                video_level_frame_number += 1

        # process visualization in parallel
        semaphore = mp.Semaphore(n_processes=10)
        semaphore.parallelize(frame_index_to_kwargs, self.demo_single_frame)
        plt.close('all')

        # stats
        duration = int(time.time()-start_time)
        print(f'done with demo frames at {video_level_frame_number/duration:.3f} FPS in total time of {duration}')

        # compile frames to video with ffmpeg
        try:
            FfmpegWrapper().compile_video(working_dir, self.sampled_fps, self.out_fps, self.demo_path)
            print(f'Saved the demo video into: {self.demo_path}')
            time.sleep(2)
            shutil.rmtree(working_dir)
        except PermissionError as e:
            eprint(f'Generating a demo video raised a PermissionError exception: {e}', e)
            eprint(f'Continue execution without deleting the visualization dir: {working_dir}')
        except Exception as e:
            eprint('Generating a demo video raised an exception.', e)
            raise e

    def demo_single_frame(self, frame_str: str, frame_path: str, shot_name: str, detections: list,
                          shot_to_offline_mot: OrderedDict, visualization_dir: str):
        """
        draw a single frame as part of the demo video
        :param frame_str: the string representation of the frame's index
        :param frame_path: the path to the frame
        :param shot_name: the shot str
        :param detections: the list of CharacterDetections from the json
        :param shot_to_offline_mot: the shot's MOT track ids
        :param visualization_dir: the output visualization repo
        :return:
        """
        fig = plt.figure(frame_str)
        ax = fig.add_subplot(111, aspect='equal')
        im = io.imread(frame_path)
        ax.imshow(im)
        ax.set_title(f"{shot_name}, {frame_str}")
        for box in detections:
            if shot_name not in shot_to_offline_mot or box.Id not in shot_to_offline_mot[shot_name]:
                continue
            track_id = shot_to_offline_mot[shot_name][box.Id]

            # draw a rectangular bbox
            rect = patches.Rectangle((box.Rect.X, box.Rect.Y), box.Rect.Width, box.Rect.Height, fill=False,
                                     lw=2.5, ec=self.colors[track_id % self.num_distinct_colors, :])
            ax.add_patch(rect)
            box_title = f'{track_id}, C:{box.Confidence:.2f}'
            ax.text(box.Rect.X, box.Rect.Y, box_title, horizontalalignment='left', verticalalignment='top', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(visualization_dir, self.frame_name_format % int(frame_str)))
        fig.canvas.flush_events()
        ax.cla()
        plt.close(frame_str)
