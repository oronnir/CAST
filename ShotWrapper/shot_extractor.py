import os
import subprocess
from E2E.configuration_loader import Configuration


config = Configuration().get_configuration()
EXE_FOLDER, EXE_NAME = os.path.split(config['shotExtractor'])
SHOTS_OUTPUT_SUFFIX = '_shot_scene.txt'
KEYFRAMES_DIR = 'Keyframes'


class Keyframe:
    def __init__(self, keyframe_name):
        self.id, self.frame_number = self.parse_image_name(keyframe_name)
        self.file = keyframe_name

    @staticmethod
    def parse_image_name(keyframe_name):
        split_args = keyframe_name.split('.jpg')[0].split('_')
        if len(split_args) < 4:
            raise Exception('bad keyframe name')

        return int(split_args[-2]), int(split_args[-1])

    def __repr__(self):
        return f'Keyframe(id:{self.id}, frame:{self.frame_number})'


class Shot:
    def __init__(self, shot_line):
        attributes_str = shot_line.split(',')
        self.id = int(attributes_str[0].split('#')[-1])
        self.start = float(attributes_str[2].split(' ')[-1])
        self.end = float(attributes_str[3].split(' ')[-1])
        frame_range = attributes_str[4].split('Frame: ')[-1].split(' ~ ')
        self.frame_start = int(frame_range[0])
        self.frame_end = int(float(frame_range[1]))
        self.keyframes = []

    def __repr__(self):
        return f'Shot(id:{self.id}, start:{self.frame_start}, end:{self.frame_end})'


class ShotExtractor:
    """
    [dump_shot_scene_list]: when specified 1, write detected shot and scene list into a txt file. Default: 0
    [scene_number]: Maximum number of scenes expected to output. Default: 30
    [dump_shot_features]: when specified 1, write extracted shot features into a txt file. Default: 0
    [dump_shot_keyframes]: when specified 1, write extracted shot keyframes to PPM image files. Default: 0
    [output_static_thumbnail_folder]:Output folder for static thumbnail images. Default: Executable location.
    [max_static_thumbnail_count]: How many static thumbnails should output? Negative number means no static thumbnail output, 0 means using default setting. Default: 0
    [max_motion_thumbnail_length]: How long the motion thumbnail shuold be? Negative number menas no motion thumbnail output, 0.0 means using default setting. Default: 0.0
    [output_audio]: when specified 1, output audio in the motion thumbnail; otherwise no. Default: 1
    [fade_in_fade_out]: when specified 1, add fade in/out effects to motion thumbnail, otherwise no. Default: 1
    """
    def __init__(self):
        self.exe_path = f'"{os.path.join(EXE_FOLDER, EXE_NAME)}"'

    @staticmethod
    def exe_cmd(input_video_path, output_path):
        return f'"{input_video_path}" "{output_path}" 1 30 1 1 1 11 1'

    def extract_shots(self, input_video_path, output_folder):
        if not os.path.isdir(output_folder):
            raise Exception('output_folder does not exist')

        shots_output_path = os.path.join(output_folder, f'{os.path.basename(input_video_path)}{SHOTS_OUTPUT_SUFFIX}')

        # lazy shot extraction
        if os.path.exists(shots_output_path):
            print('shots have already been extracted - skipping execution...')
            shots = self.parse(shots_output_path)
            return shots

        # execute
        print('start extracting shots')
        os.chdir(output_folder)
        result = subprocess.run(' '.join([self.exe_path, self.exe_cmd(input_video_path, output_folder)]),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        output, err = result.stdout, result.stderr
        exit_code = result.returncode

        # validate
        if exit_code == 0:
            print("finished extracting shots")
        else:
            raise Exception(f'failed extracting shots with exit code {exit_code} and error: {err}')

        # parse shots
        if not os.path.exists(shots_output_path):
            raise Exception('failed extracting shots')
        shots = self.parse(shots_output_path)
        return shots

    @staticmethod
    def parse(txt_path):
        with open(txt_path, 'r') as out_file:
            lines = out_file.readlines()
        shots = []
        for line in lines:
            if not line.startswith('Shot #'):
                continue
            shots.append(Shot(line))
        return shots

    @staticmethod
    def __match_keyframes_to_shots(kfs, shots):
        kfs = sorted(kfs, key=lambda kf: kf.frame_number)
        shots = sorted(shots, key=lambda sh: sh.frame_start)
        k = 0
        s = 0
        while k < len(kfs) and s < len(shots):
            if shots[s].frame_start <= kfs[k].frame_number <= shots[s].frame_end:
                shots[s].keyframes.append(kfs[k])
                k += 1
            else:
                s += 1
        return shots
