import json
import os
import sys
import subprocess
import tempfile

from Animator.bbox_grouper_api import CharacterDetectionOutput
from FfmpegWrapper.ffmpeg_wrapper import FfmpegWrapper
from E2E.configuration_loader import Configuration


config = Configuration().get_configuration()
EXE_FOLDER, EXE_NAME = os.path.split(config['detector'])
CONFIG = config['detectorConfig']
DETECTION_JSON_NAME = 'animationdetectionoutput.json'


class DetectorWrapper:
    """
    A CMD wrapper for the Yolo detector.
    """
    def __init__(self):
        self.exe_path = os.path.join(EXE_FOLDER, EXE_NAME)
        if not os.path.isfile(self.exe_path):
            raise Exception(f'Exe file does not exist at: {self.exe_path}')
        self.config_path = CONFIG
        if not os.path.isfile(self.config_path):
            raise Exception(f'Config file does not exist at: {self.config_path}')
        self.ffmpeg = FfmpegWrapper()

    def detect_and_featurize(self, images_dir, output_path, multi_repo_json_path=None):
        if not multi_repo_json_path:
            if not os.path.isdir(images_dir):
                raise Exception(f'Input dir does not exist: "{images_dir}"')
            if not os.path.isdir(output_path):
                raise Exception(f'Output dir does not exist: "{output_path}"')
            os.chdir(output_path)

        print('start detecting characters')
        cmd = ' '.join(self._command_line(images_dir, output_path, multi_repo_json_path))
        result = subprocess.run(cmd,
                                stdout=sys.stdout,
                                stderr=subprocess.PIPE,
                                universal_newlines=True)

        output, err = result.stdout, result.stderr
        exit_code = result.returncode

        if exit_code == 0:
            print("finished detecting characters")
        else:
            raise Exception(f'failed detecting boxes with exit code {exit_code} and error: {err}')
        output_json_path = os.path.join(output_path, DETECTION_JSON_NAME)
        if not multi_repo_json_path and not os.path.isfile(output_json_path):
            raise Exception(f'Output json does not exist: "{output_json_path}"')
        return CharacterDetectionOutput.read_from_json(output_json_path)

    def detect_multi_repo(self, input_repos: list, output_repos: list):
        """
        run the detector+featurizer on multiple repositories without loading the model multiple times
        :param input_repos: the input list of repositories containing shots frames
        :param output_repos: the output directory
        :return: None
        """
        if len(input_repos) != len(output_repos):
            raise Exception('Invalid input with inconsistent input/output repos sizes')

        if len(input_repos) == 0:
            return

        # skip the processed repos
        input_repos = [input_repos[i] for i in range(len(input_repos))
                       if not os.path.isfile(os.path.join(output_repos[i], DETECTION_JSON_NAME))]
        output_repos = [output_repos[i] for i in range(len(output_repos))
                        if not os.path.isfile(os.path.join(output_repos[i], DETECTION_JSON_NAME))]

        # in case all were processed then return
        if len(input_repos) == 0:
            return

        # work on the missing files
        json_path = os.path.join(tempfile.gettempdir(), 'detect_multi.json')
        if os.path.isfile(json_path):
            os.remove(json_path)
        with open(json_path, 'w') as j_writer:
            json.dump({"Inputs": input_repos, "Outputs": output_repos}, j_writer)

        # apply the detection
        self.detect_and_featurize("", "", json_path)

    def extract_detect_feat(self, video_path, frames_dir, detections_dir):
        fps = 3
        self.ffmpeg.extract_frames(video_path, frames_dir, fps)
        self.detect_and_featurize(frames_dir, detections_dir)

    def _command_line(self, images_dir, output_dir, multi_repo_json_path=None):
        command = f'"{images_dir}" "{output_dir}" "{self.config_path}" 3'
        if multi_repo_json_path:
            command += f' -m "{multi_repo_json_path}"'
        return f'"{self.exe_path}"', command
