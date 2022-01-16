"""
The input data for the fine-tune flow. Directory structure in "user@118:/home/students/oron/animation/data$"

data:
	\TRAIN_Series:
		\triplets:
			\guid.jpg
			\triplets.json
		\model:
		\keyframe_detections:
			\guid.jpg
			\animationdetectionoutput.json
"""

import json
import os
import shutil
import uuid
from typing import List

from Featurizer.featurizer import Featurizer
from Tracker.tracks import Triplet
from Utils.gpu_profiler import nvidia_smi
from Animator.consolidation_api import CharacterDetectionOutput
from Animator.utils import create_dir_if_not_exist


def fine_tune_featurizer(image_repo: str, triplets: List[Triplet], config: dict, output_model_repo: str) -> Featurizer:
    """run tuning and serialize"""
    output_model_path = os.path.join(output_model_repo, f'ft_{config["session_id"]}.pth')
    base_model = Featurizer.get_base_model(config['gpu_id'], config["session_id"], output_model_path)
    base_model.fine_tune(image_repo, triplets, config)
    # now the base model is finetuned
    return base_model


def evaluate_featurizer(test_data, model):
    pass


def load_eval_data(episode_image_repo: str, detections: CharacterDetectionOutput, featurizer: Featurizer, gpu_id: str,
                   num_workers: int):
    """re-embed all detections in the training episode and run the grouper flow"""
    # featurize the detections folder into a dict
    feature_vec, image_paths = featurizer.featurize(episode_image_repo, num_workers=num_workers, gpu_id=gpu_id)
    image_ids = [os.path.basename(p).rsplit('.')[0] for p in image_paths]
    thumbnail_to_features = dict([(thumbnail_id, features) for (thumbnail_id, features) in zip(image_ids, feature_vec)])

    # populate the features in-place
    for box in detections.CharacterBoundingBoxes:
        image_path = os.path.join(episode_image_repo, box.ThumbnailId + '.jpg')
        if not os.path.isfile(image_path) or box.ThumbnailId not in thumbnail_to_features:
            print(f'cannot find {image_path}')
            box.Features = None
            continue
        box.Features = thumbnail_to_features[box.ThumbnailId]
    return detections


def fine_tune_flow(video_root: str, gpu_id: str, num_workers: int) -> Featurizer:
    # generate a correlation id
    session_id = str(uuid.uuid4())
    episode_name = os.path.basename(video_root)
    print(f'starting a fine-tune session: {session_id}')

    # load triplets
    triplets_repo = os.path.join(video_root, r'triplets')
    triplets_json = os.path.join(triplets_repo, 'triplets.json')
    output_model_repo = create_dir_if_not_exist(triplets_repo, 'model')
    with open(triplets_json, 'r') as t_reader:
        triplets = json.load(t_reader)

    # fine tune
    start_epoch = 41
    additional_epochs = 10
    config = dict(session_id=session_id, gpu_id=gpu_id, workers=num_workers, batch_size=20,
                  epochs=start_epoch + additional_epochs, margin=1., episode=episode_name, gamma=0.1, momentum=0.9,
                  lr=2e-5, start_epoch=start_epoch, optimizer='AdamW', weight_decay=1e-4)
    tuned_model = fine_tune_featurizer(triplets_repo, triplets, config, output_model_repo)
    return tuned_model


def embed_detections(tuned_model: Featurizer, gpu_id: str, num_workers: int, episode_name: str) -> None:
    # evaluate the model
    print(f'start embedding the detections with the finetuned featurizer on episode: {episode_name}')
    training_episode_repo = f'/../SAIL/{episode_name}/keyframe_detections/animationdetectionoriginalimages'
    old_det_suffix = '_old'  # used to be '_copy'
    detection_json_path = os.path.join(training_episode_repo, '..', f'animationdetectionoutput{old_det_suffix}.json')
    detections = CharacterDetectionOutput.read_from_json(detection_json_path)
    if detections is None:
        print(f'Failed loading the detections json! Check: {detection_json_path}')

    test_dataset = load_eval_data(training_episode_repo, detections, tuned_model, gpu_id, num_workers)
    output_json_path = detection_json_path.replace(f'{old_det_suffix}.json', '.json')
    test_dataset.save_as_json(output_json_path)
    print(f'serialized detections into {output_json_path}')


def remote_triplets_loading():
    pass


def main_fine_tune():
    nvidia_smi()
    video_root_paths = [
        # our Evaluation
        # r'/../TRAINING_Land_Before_Time-Return_To_Hanging_Rock',
        # r'/../TRAINING_Southpark',
        # r'/../TRAINING_FiremanSam-FireBelow',
        # r'/../TRAIN_Floogals',
        # r'/../TRAIN_FairlyOddParents',
        # r'/../TRAIN_Garfield',
        # r'/../TRAIN_BobTheBuilder',
        # r'/../TRAIN_DextersLab'

        # SAIL imdb
        # r'/../SAIL/FreeBirds',
        # r'/../SAIL/Shrek4',
        # r'/../SAIL/Frozen',
        # r'/../SAIL/Dragon',
        # r'/../SAIL/Cars',
        # r'/../SAIL/Tangled',
        # r'/../SAIL/ToyStory',
        r'/../SAIL/Lego',
    ]

    for video_root_path in video_root_paths:
        print(f'Start analyzing video: {video_root_path}')
        if not os.path.isdir(video_root_path):
            raise Exception(f'Directory do not exist: {video_root_path}')

        episode = os.path.basename(video_root_path)
        gpu = '0'
        n_workers = 5

        # rename the detection file
        current_detection_folder = os.path.join(video_root_path, 'keyframe_detections')
        past_detections_json_to_rename = os.path.join(current_detection_folder, 'animationdetectionoutput.json')
        old_detections_json = os.path.join(current_detection_folder, 'animationdetectionoutput_old.json')

        # fix name error
        if os.path.isfile(
                os.path.join(current_detection_folder, 'old_animationdetectionoutput.json')) and not os.path.isfile(
            old_detections_json):
            shutil.copy(os.path.join(current_detection_folder, 'old_animationdetectionoutput.json'),
                        old_detections_json)

        if os.path.isfile(past_detections_json_to_rename):
            os.rename(past_detections_json_to_rename, old_detections_json)
        if not os.path.isfile(old_detections_json):
            raise Exception(f'Old detections are not copied properly to: {old_detections_json}')

        # collage_flow = False
        # if collage_flow:
        #     local_video_root = r"\..\TRAINING_Land Before Time - Return To Hanging Rock"
        #     video_track_pkl_path = os.path.join(local_video_root, r"mot\video_track_external_TRAINING_Land Before Time - Return To Hanging Rock.mp4.pkl")
        #     local_triplet_sampling(video_track_pkl_path, local_video_root)

        # run finetune flow
        # model_path = os.path.join(video_root_path, 'triplets/model/ft_41732257-59bd-47b3-b28e-517117c2b01c41732257.pth')
        # ft_featurizer = Featurizer.get_finetuned_model(gpu, model_path)
        # model_path = os.path.join(video_root_path, 'triplets/model/ft_84df0a01-437c-4e0a-bfad-8d52b7a51cb284df0a01.pth')
        # ft_featurizer = Featurizer.get_finetuned_model(gpu, model_path)
        ft_featurizer = fine_tune_flow(video_root_path, gpu, n_workers)

        # embed detections
        embed_detections(ft_featurizer, gpu, n_workers, episode)
        print('Done!')


if __name__ == '__main__':
    main_fine_tune()
