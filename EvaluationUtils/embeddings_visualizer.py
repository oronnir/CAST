from Animator.bbox_grouper_api import CharacterDetectionOutput
import os
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf


def project_single_repo(repo, characters, name_to_cluster, name_to_series):
    series_name = os.path.basename(repo).split('_')[0]
    detections_json = os.path.join(repo, 'animationdetectionoutput.json')
    clusters_dir = os.path.join(repo, 'Clusters')
    clusters_names = os.listdir(clusters_dir)

    # parse the detections
    name_to_detections = dict()
    characters_detections = CharacterDetectionOutput.read_from_json(detections_json)
    for det in characters_detections.CharacterBoundingBoxes:
        name_to_detections[f'{det.ThumbnailId}.jpg'] = det.Features

    # keep vectors per image
    for cluster_name in clusters_names:
        clusters_dir_path = os.path.join(clusters_dir, cluster_name)
        cluster_images = os.listdir(clusters_dir_path)
        for image_name in cluster_images:
            image_path = os.path.join(clusters_dir_path, image_name)
            if image_name not in name_to_detections:
                os.remove(image_path)
                continue
            characters[image_path] = name_to_detections[image_name]
            name_to_cluster[image_name] = cluster_name
            name_to_series[image_name] = series_name

    return characters, name_to_cluster, name_to_series


def project_multi_series(root_dir, thumbnail_size):
    # clear output
    embeddings_json_path = os.path.join(root_dir, 'embeddings.json')
    if os.path.isfile(embeddings_json_path):
        os.remove(embeddings_json_path)
    images_paths_txt = os.path.join(root_dir, 'images.txt')
    if os.path.isfile(images_paths_txt):
        os.remove(images_paths_txt)

    # loop over all repos and aggregate data
    characters = dict()
    name_to_cluster = dict()
    name_to_series = dict()
    root_content = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
    repos = [d for d in root_content if os.path.isdir(d)]
    for repo_path in repos:
        print(f'Start analyzing repo: {repo_path}')
        characters, name_to_cluster, name_to_series = project_single_repo(repo_path, characters, name_to_cluster,
                                                                          name_to_series)

    # serialize results
    vectors_tsv = os.path.join(root_dir, 'vectors.tsv')
    ordered_vectors = []
    sprite_order = []
    with open(vectors_tsv, 'w') as v_tsv:
        for p, v in characters.items():
            sprite_order.append(p)
            ordered_vectors.append(v)
            v_tsv.write('\t'.join([str(f) for f in v]))
            v_tsv.write('\n')

    with open(images_paths_txt, 'w') as i_txt:
        i_txt.write('\n'.join([s for s in sprite_order]))

    # tutorial
    width, height = thumbnail_size, thumbnail_size
    images = [Image.open(filename).resize((width, height)) for filename in sprite_order]
    image_width, image_height = images[0].size
    one_square_size = int(np.ceil(np.sqrt(len(images))))
    master_width = (image_width * one_square_size)
    master_height = image_height * one_square_size
    spriteimage = Image.new(
        mode='RGBA',
        size=(master_width, master_height),
        color=(0, 0, 0, 0))  # fully transparent

    for count, image in enumerate(images):
        div, mod = divmod(count, one_square_size)
        h_loc = image_width * div
        w_loc = image_width * mod
        spriteimage.paste(image, (w_loc, h_loc))

    sprite_path = os.path.join(root_dir, 'sprite.jpg')
    spriteimage.convert("RGB").save(sprite_path, transparency=0)
    existing_images_df = pd.DataFrame(data={
        'series': [name_to_series[os.path.basename(f)] for f in sprite_order],
        'cat_id': [name_to_cluster[os.path.basename(f)] for f in sprite_order],
        'pid': [f for f in sprite_order]
    })

    metadata_path = os.path.join(root_dir, 'metadata.tsv')
    metadata = existing_images_df[['series', 'cat_id', 'pid']] \
        .to_csv(metadata_path, sep='\t', index=False)

    print(f'Done building resources for {len(sprite_order)} bboxes of {len(set(name_to_series.values()))} series')

    from tensorboard.plugins.projector.projector_config_pb2 import ProjectorConfig
    config = ProjectorConfig()
    from tensorboard.plugins import projector

    # Create randomly initialized embedding weights which will be trained.
    data = np.asarray(ordered_vectors)
    embedding_var = tf.Variable(data, name=os.path.basename(root_dir))

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.tensor_path = vectors_tsv

    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(root_dir, 'metadata.tsv')

    embedding.sprite.image_path = sprite_path

    # Specify the width and height of a single thumbnail.
    embedding.sprite.single_image_dim.extend([width, height])

    # Use the same LOG_DIR where you stored your checkpoint.
    summary_writer = tf.summary.create_file_writer(root_dir)

    # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
    # read this file during startup.
    projector.visualize_embeddings(root_dir, config)


if __name__ == '__main__':
    stop = 0
    thumbnails_pixels = 55
    all_series = r'\..\Figures\Embeddings\TensorBoardProjector2'
    project_multi_series(all_series, thumbnails_pixels)
    stop2 = 1  # run on CMD: tensorboard --logdir <all_series>
