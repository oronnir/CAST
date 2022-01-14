import os
import subprocess
import cv2
from subprocess import PIPE, run
from Animator.utils import eprint, colored
from E2E.configuration_loader import Configuration


config = Configuration().get_configuration()
FFMPEG_EXE_LOCATION = config['ffmpeg']
FFPROBE_EXE_LOCATION = config['ffprobe']


class FfmpegWrapper:
    def __init__(self):
        if not os.path.isfile(FFMPEG_EXE_LOCATION):
            raise Exception('FFMPEG EXE not found')
        self._ffmpeg = FfmpegWrapper._quote(FFMPEG_EXE_LOCATION)
        self._ffprobe = FfmpegWrapper._quote(FFPROBE_EXE_LOCATION)

    def extract_frames(self, video_path, output_folder, fps, frame_range=None):
        if not os.path.isdir(output_folder):
            raise Exception('output folder does not exist')
        if not os.path.isfile(video_path):
            raise Exception('input video does not exist')
        if frame_range:
            cv2_extract_frames(video_path, output_folder, frame_range)
            return
        cmd = f'{self._ffmpeg} -i "{video_path}" -vf fps={fps} "{output_folder}\\%6d.jpg"'
        exit_code = subprocess.call(cmd)
        if exit_code != 0:
            raise Exception(f'failed extracting frames with exit code {exit_code}')

    def compile_video(self, frames_path: str, sampled_fps: float, out_fps: float, output_video_path: str) -> None:
        if not os.path.isdir(frames_path):
            raise Exception('output folder does not exist')
        os.chdir(frames_path)
        cmd = f'-y -framerate {sampled_fps} -i %06d.jpg -c:v libx264 -r {out_fps} -pix_fmt yuv420p "{output_video_path}"'
        result = run(' '.join([self._ffmpeg, cmd]), stdout=PIPE, stderr=PIPE, universal_newlines=True)
        output, err = result.stdout, result.stderr
        exit_code = result.returncode
        if exit_code != 0:
            eprint(err)
            raise Exception(f'failed generating video with exit code {exit_code} and error: {err}')
        print(colored(f'successfully compiled a video into: {output_video_path}', 'green'))

    def get_frame_rate(self, video_path):
        cmd = f' -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate "{video_path}"'
        output, error = FfmpegWrapper._verify_input_and_execute(video_path, self._ffprobe, cmd, 'FPS')
        fps_numerator, fps_denominator = str(output).strip().split("/")
        return float(fps_numerator) / int(fps_denominator)

    @staticmethod
    def _verify_input_and_execute(video_path, exe_path, cmd, message):
        if not os.path.isfile(video_path):
            raise Exception('input video does not exist')
        result = run(' '.join([exe_path, cmd]), stdout=PIPE, stderr=PIPE, universal_newlines=True)
        output, err = result.stdout, result.stderr
        exit_code = result.returncode
        if exit_code != 0:
            eprint(err)
            raise Exception(f'failed getting {message} with exit code {exit_code} and error: {err}')
        return output, err

    def get_frame_dimensions(self, video_path):
        cmd = f' -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "{video_path}"'
        output, error = FfmpegWrapper._verify_input_and_execute(video_path, self._ffprobe, cmd, 'frame dimensions')
        w, h = str(output).strip().split(",")
        return int(w), int(h)

    @staticmethod
    def _quote(path):
        if '"' in path or "'" in path:
            return path
        return f'"{path}"'


def cv2_extract_frames(video_path, output_folder, frame_range):
    start, end = frame_range
    cap = cv2.VideoCapture(video_path)

    # get total number of frames
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # check for valid frame number
    if end >= 0 & end <= total_frames:
        # set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_counter = 1
    while frame_counter <= end - start:
        ret, frame = cap.read()
        frame_path = os.path.join(output_folder, f'{frame_counter:06d}.jpg')
        cv2.imwrite(frame_path, frame)  # save frame as JPEG file
        frame_counter += 1
