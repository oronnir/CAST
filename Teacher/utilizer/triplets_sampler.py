import os
from shutil import copyfile, make_archive
from typing import List

from EvaluationUtils.descriptive_stats import create_collage
from Tracker.tracks import VideoTrack, Triplet
from Animator.utils import create_dir_if_not_exist, to_json, colored

triplets_repo_name = 'triplets'


def generate_contrastive_triplets(video_track: VideoTrack) -> List[Triplet]:
    n = 10000
    normalized_distance_per_second = 3.
    fade_frames = 15
    triplets = video_track.sample_triplets(n, normalized_distance_per_second, fade_frames)
    return triplets


def local_triplet_sampling(pkl_path: str, root_repo: str):
    triplet_im_repo = create_dir_if_not_exist(root_repo, triplets_repo_name)
    triplets_json = os.path.join(triplet_im_repo, 'triplets.json')
    if os.path.isfile(triplets_json):
        print(colored('Triplets json already exists! Skipping execution...', 'yellow'))
        return

    video_track = VideoTrack.deserialize(pkl_path)

    # generate a triplet-loss based training dataset
    triplets = generate_contrastive_triplets(video_track)
    print('Start copying triplets...')

    # create a local triplets repo
    collage_path = create_dir_if_not_exist(root_repo, 'collages')
    for i, triplet in enumerate(triplets):
        shot_det_repo = os.path.join(root_repo, 'video_shots', triplet.shot_id, 'det')
        source_paths = dict(anchor=os.path.join(shot_det_repo, f'{triplet.anchor}.jpg'),
                            negative=os.path.join(shot_det_repo, f'{triplet.negative}.jpg'),
                            positive=os.path.join(shot_det_repo, f'{triplet.positive}.jpg'))
        for triplet_type, src in source_paths.items():
            dest = os.path.join(triplet_im_repo, os.path.basename(src))
            copyfile(src, dest)

        if i % 50 != 0:
            continue
        collage_name = f'Triplet_{triplet.anchor[:6]}_{triplet.positive[:6]}_{triplet.negative[:6]}.jpg'
        try:
            create_collage(source_paths.values(), os.path.join(collage_path, collage_name), source_paths.keys())
        except Exception as e:
            print(e)

    # serialize triplets
    to_json(triplets, triplets_json)
    print(f'Done with triplets... Serialized to: {triplets_json}')

    # zip triplets+json to transfer into the next stage...
    video_name = os.path.basename(root_repo).replace(' ', '_')
    cwd = os.getcwd()
    os.chdir(root_repo)
    make_archive(f'full-{triplets_repo_name}-{video_name}', 'zip', triplet_im_repo)
    print('Triplets repo is zipped!')
    os.chdir(cwd)
    return
