import os
import shutil

from Animator.bbox_grouper_api import CharacterDetectionOutput
from Animator.utils import create_dir_if_not_exist, try_zip_folder

det_json_name = 'animationdetectionoutput.json'


def local_prepare_data_for_finetune(series_role_folder, remote_folder_name, triplets_local_dir):
    zip_path = os.path.join(series_role_folder, remote_folder_name + '.zip')
    if os.path.isfile(zip_path):
        return

    original_det_json_path = os.path.join(series_role_folder, det_json_name)
    if not os.path.isdir(series_role_folder) \
            or not os.path.isfile(original_det_json_path)\
            or not os.path.isdir(triplets_local_dir):
        raise Exception('Wrong path!')

    dump_dir = create_dir_if_not_exist(series_role_folder, remote_folder_name)
    detections_dir = create_dir_if_not_exist(dump_dir, 'keyframe_detections')
    detection_json_destination = os.path.join(detections_dir, det_json_name)
    if not os.path.isfile(detection_json_destination):
        character_detections = CharacterDetectionOutput.read_from_json(original_det_json_path)
        for det in character_detections.CharacterBoundingBoxes:
            det.Features = []

        character_detections.save_as_json(detection_json_destination)

    source_detections = os.path.join(series_role_folder, 'animationdetectionoriginalimages')
    destination_detections = os.path.join(detections_dir, 'animationdetectionoriginalimages')
    if not os.path.isdir(destination_detections):
        shutil.copytree(source_detections, destination_detections)

    triplets_dump_path = os.path.join(dump_dir, 'triplets')
    if not os.path.isdir(triplets_dump_path):
        shutil.copytree(triplets_local_dir, triplets_dump_path)

    return try_zip_folder(dump_dir, zip_path)


if __name__ == '__main__':
    sers_to_trips = {
        'BobTheBuilder': r'..\Training_BobTheBuilder\triplets',
        'FairlyOddParents': r'..\TRAINING_FairlyOddParents\triplets',
        'FiremanSam': r'..\TRAINING_FiremanSam\triplets',
        'Floogals': r'..\TRAINING_FLOOGALS\triplets',
        'Garfield': r'..\TRAINING_Garfield\triplets',
        'Southpark': r'..\TRAINING_Southpark\triplets',
        'The Land Before Time': r'..\TRAINING_The Land Before Time\triplets'
    }
    root_path = r'...\E2ETestset\TripletsSeNet'
    for ser, trip_dir in sers_to_trips.items():
        print(f'start preparing data for series: {ser}')
        role_path = os.path.join(root_path, ser, 'Training')
        remote_name = f"TRAIN_{ser.replace(' ', '_')}"
        local_prepare_data_for_finetune(role_path, remote_name, trip_dir)
        print(f'done preparing data for series: {ser}')
