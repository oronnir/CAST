import os
import tempfile
import json
from tqdm import tqdm
from shutil import copytree, rmtree, copyfile, make_archive
from Teacher.utilizer.triplets_sampler import triplets_repo_name
from Animator.consolidation_api import CharacterDetectionOutput
from Animator.utils import create_dir_if_not_exist
from E2E.main_1 import detection_file_name


def anchors_to_json(vid_path):
    tmp = tempfile.gettempdir()
    work_dir_name = os.path.basename(vid_path).split('.')[0]
    working_repo_path = os.path.join(tmp, work_dir_name)
    if not os.path.isdir(working_repo_path):
        raise Exception(f'Dir not found at: {working_repo_path}!')
    finetune_dir = create_dir_if_not_exist(working_repo_path, work_dir_name)
    detections_dir = create_dir_if_not_exist(finetune_dir, 'keyframe_detections')
    anim_det_orig_imgs = create_dir_if_not_exist(detections_dir, 'animationdetectionoriginalimages')

    # copy triplets to the new location
    old_triplets_repo_path = os.path.join(working_repo_path, triplets_repo_name)
    new_triplets_repo_path = os.path.join(finetune_dir, triplets_repo_name)
    if not os.path.isdir(new_triplets_repo_path):
        copytree(old_triplets_repo_path, new_triplets_repo_path)
        rmtree(old_triplets_repo_path)

    # load the triplets
    triplets_json = os.path.join(new_triplets_repo_path, 'triplets.json')
    with open(triplets_json, 'r') as j_reader:
        triplets = json.load(j_reader)

    # sort by shot id
    triplets = sorted(triplets, key=lambda t: int(t['shotId'].split('_')[1]))

    shots_repo = os.path.join(working_repo_path, 'video_shots')
    character_detection_output = CharacterDetectionOutput(dict(characterBoundingBoxes=[]))
    current_shot_detections = None
    prev_shot_id = -1
    new_id = 1

    for triplet in tqdm(triplets):
        thumbnail_id = triplet['anchor']
        shot_name = triplet['shotId']
        if shot_name != prev_shot_id:
            shot_repo_path = os.path.join(shots_repo, shot_name)
            current_shot_det_path = os.path.join(shot_repo_path, 'det', detection_file_name)
            current_shot_detections = CharacterDetectionOutput.read_from_json(current_shot_det_path)
        matching_tid = [d for d in current_shot_detections.CharacterBoundingBoxes if d.ThumbnailId == thumbnail_id]
        if len(matching_tid) is not 1:
            print(f'Got {len(matching_tid)} thumbnail ids with thumbnailid: {thumbnail_id} on shot {shot_name}. Skipping!')
            continue

        anc_det = matching_tid[0]
        if not os.path.isfile(anc_det.File):
            print(f'Could not find file: {anc_det.File}. Skipping!')
            continue

        anc_det.Features = []
        anc_det.Id = new_id
        character_detection_output.CharacterBoundingBoxes.append(anc_det)
        new_id += 1
        character_detection_output.NativeKeyframeHeight = current_shot_detections.NativeKeyframeHeight
        character_detection_output.NativeKeyframeWidth = current_shot_detections.NativeKeyframeWidth

        # copy the jpg
        file_name = os.path.basename(anc_det.File)
        destination_file_path = os.path.join(anim_det_orig_imgs, file_name)
        if not os.path.isfile(destination_file_path):
            copyfile(anc_det.File, destination_file_path)

    # serialize
    output_json_path = os.path.join(detections_dir, f'old_{detection_file_name}')
    character_detection_output.save_as_json(output_json_path)
    print(f'Success serialized {new_id} files.')

    # zip triplets+json to transfer into the next stage...
    cwd = os.getcwd()
    os.chdir(working_repo_path)
    make_archive(f'{triplets_repo_name}-{work_dir_name}', 'zip', finetune_dir)
    print('Triplets repo is zipped!')
    os.chdir(cwd)


if __name__ == '__main__':
    video_path = r"..\SAIL_animation_movie_character_database\videos\The Lego Movie (2014) [1080p]\TheLegoMovie_s1280.mp4"
    anchors_to_json(video_path)
