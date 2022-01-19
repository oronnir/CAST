import os
import shutil
import json
import math
from EvaluationUtils.detection_mapping import DetectionMapping
from Animator.utils import eprint
from EvaluationUtils.image_utils import crop_image, save_image
from Animator.consolidation_api import CharacterDetectionOutput
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw
import time
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def find_similar_and_dissimilar_pairs(num_examples, ids, features):
    n = len(ids)
    pairs = list()
    distances = list()
    for i in tqdm(range(n)):
        for j in range(i+1, n):
            diff = features[i] - features[j]
            diff_square = diff.T.dot(diff)
            l2_norm = math.sqrt(diff_square)
            pairs.append([ids[i], ids[j], l2_norm])
            distances.append(l2_norm)
    distances = np.asarray(distances)
    linear_top_dis = np.argpartition(distances, -num_examples)[-num_examples:]
    linear_top_sim = np.argpartition(distances, num_examples)[:num_examples]

    top_similar = [pairs[i] for i in linear_top_sim]
    top_dis_similar = [pairs[i] for i in linear_top_dis]
    return top_similar, top_dis_similar


def visualize_similarity_features(sim_repo, pairs, role_detections_repo):
    if os.path.isdir(sim_repo):
        shutil.rmtree(sim_repo)
    os.mkdir(sim_repo)
    counter = 0
    for sim_pair in pairs:
        pair_repo = os.path.join(sim_repo, '', str(counter))
        counter += 1
        os.mkdir(pair_repo)
        target_bbox_1 = os.path.join(pair_repo, '', '{}.jpg'.format(sim_pair[0]))
        source_bbox_1 = os.path.join(role_detections_repo, '', '{}.jpg'.format(sim_pair[0]))
        target_bbox_2 = os.path.join(pair_repo, '', '{}.jpg'.format(sim_pair[1]))
        source_bbox_2 = os.path.join(role_detections_repo, '', '{}.jpg'.format(sim_pair[1]))
        if os.path.isfile(source_bbox_1) and os.path.isfile(source_bbox_2):
            shutil.copyfile(source_bbox_1, target_bbox_1)
            shutil.copyfile(source_bbox_2, target_bbox_2)


def count_files_in_repo(repo):
    if not os.path.isdir(repo):
        return -1
    return len(os.listdir(repo))


def create_collage(source_images, target_image_path, texts=None):
    if texts is None:
        texts = [str(t) for t in range(len(source_images))]
    type_to_source_paths = dict(zip(texts, source_images))

    width, height = 1600, 900
    n = len(source_images)
    edge_count = int(math.sqrt(n)) + 1 if int(math.sqrt(n)) ** 2 < n else int(math.sqrt(n))
    cols = edge_count
    rows = edge_count
    thumbnail_width = width//cols
    thumbnail_height = height//rows
    size = thumbnail_width, thumbnail_height
    new_im = Image.new('RGB', (width, height))
    ims = []
    for triplet_type, p in type_to_source_paths.items():
        im = Image.open(p)
        im.thumbnail(size)

        # write label
        draw = ImageDraw.Draw(im)
        draw.text((0, 0), triplet_type, (255, 255, 255))
        ims.append(im)
    i = 0
    x = 0
    y = 0
    for col in range(cols):
        for row in range(rows):
            if n == 0:
                break
            n -= 1
            new_im.paste(ims[i], (x, y))
            i += 1
            y += thumbnail_height
        x += thumbnail_width
        y = 0

    if os.path.isfile(target_image_path):
        os.remove(target_image_path)
    new_im.save(target_image_path)
    return


def deduplication_threshold_setting(series, eval_root):
    dissimilar = []
    similar = []
    deduper_repo = r'\..\Deduper'
    similar_repo = os.path.join(deduper_repo, '', 'Similar')
    dissimilar_repo = os.path.join(deduper_repo, '', 'Dissimilar')
    for ser in series:
        for role in ['Training', 'Test']:
            ser_path = os.path.join(eval_root, '', ser)
            role_path = os.path.join(ser_path, '', role)
            detection_output_path = os.path.join(role_path, '', 'animationdetectionoutput.json')

            # adding EDH features
            edh_detection_json = detection_output_path.replace(r'E2ETestset\SeResNext\Videos', r'E2ETestset\EDH')
            edh_character_detections = CharacterDetectionOutput.read_from_json(edh_detection_json)
            id_to_edh = {bbox.ThumbnailId: bbox.Features for bbox in edh_character_detections.CharacterBoundingBoxes}

            ser_similar_repo_path = os.path.join(similar_repo, ser, role)
            if os.path.isdir(ser_similar_repo_path):
                pairs_repo_names = os.listdir(ser_similar_repo_path)
                for pair_repo_name in pairs_repo_names:
                    pair_repo_path = os.path.join(ser_similar_repo_path, pair_repo_name)
                    similar_thumbs = os.listdir(pair_repo_path)
                    if len(similar_thumbs) <= 1:
                        continue
                    first_thumbnail_id = similar_thumbs[0].replace('.jpg', '')
                    second_thumbnail_id = similar_thumbs[1].replace('.jpg', '')
                    if first_thumbnail_id in id_to_edh and second_thumbnail_id in id_to_edh:
                        cos = cosine_similarity([id_to_edh[first_thumbnail_id], id_to_edh[second_thumbnail_id]])[0, 1]
                        print('Similar: first thumb: {}, second thumb: {}, cosine: {}'.format(first_thumbnail_id,
                                                                                              second_thumbnail_id, cos))
                        similar.append(cos)

            ser_dissimilar_repo_path = os.path.join(dissimilar_repo, ser, role)
            if os.path.isdir(ser_dissimilar_repo_path):
                pairs_repo_names = os.listdir(ser_dissimilar_repo_path)
                for pair_repo_name in pairs_repo_names:
                    pair_repo_path = os.path.join(ser_dissimilar_repo_path, pair_repo_name)
                    dissimilar_thumbs = os.listdir(pair_repo_path)
                    if len(dissimilar_thumbs) <= 1:
                        continue
                    first_thumbnail_id = dissimilar_thumbs[0].replace('.jpg', '')
                    second_thumbnail_id = dissimilar_thumbs[1].replace('.jpg', '')
                    if first_thumbnail_id in id_to_edh and second_thumbnail_id in id_to_edh:
                        cos = cosine_similarity([id_to_edh[first_thumbnail_id], id_to_edh[second_thumbnail_id]])[0, 1]
                        print('disSimilar: first thumb: {}, second thumb: {}, cosine: {}'.format(first_thumbnail_id,
                                                                                                 second_thumbnail_id,
                                                                                                 cos))
                        dissimilar.append(cos)

    print('Similar\n{}'.format(similar))
    print('DisSimilar\n{}'.format(dissimilar))
    plt.hist(similar, bins=50, label='A complete Duplication', alpha=0.5)
    plt.hist(dissimilar, bins=50, label='Very close instances', alpha=0.5)
    plt.axvline(x=0.995, label='Merge threshold', color='r', linestyle='dashed', linewidth=1)
    plt.legend(loc='best')
    plt.show()
    plt.savefig(r'\..\Deduper\Deduplication threshold.png')
    return similar, dissimilar


def main():
    # get descriptive statistics
    eval_root = r'\..\SeResNext\Videos'
    series = os.listdir(eval_root)

    for ser in series:
        for role in ['Training', 'Test']:
            if ser not in ['FiremanSam'] or role in ['Training']:
                print('skipping {} {}'.format(ser, role))
                continue

            ser_path = os.path.join(eval_root, '', ser)
            role_path = os.path.join(ser_path, '', role)
            detection_output_path = os.path.join(role_path, '', 'animationdetectionoutput.json')

            role_detections_repo = os.path.join(role_path, '', 'animationdetectionoriginalimages')
            role_detections_count = count_files_in_repo(role_detections_repo)
            print('Series: {}, Role: {}, Count: {}'.format(ser, role, role_detections_count))

            if role_detections_count <= 0:
                print('*** SKIP - Got no detections for {} ***'.format(role_path))
                continue

            features = list()
            ids = list()
            character_detections = CharacterDetectionOutput.read_from_json(detection_output_path)

            grouping_output_path = os.path.join(role_path, '', 'animationgroupingoutput.json')
            mapping = DetectionMapping.parse_index(detection_output_path, grouping_output_path)

            # serialize mapping
            mapping_serialization_path = os.path.join(role_path, '', 'CombinedGroupedDetections.json')
            if not os.path.isfile(mapping_serialization_path):
                mapping_dict = dict(boxes=[bmap.__dict__ for bmap in mapping])
                try:
                    with open(mapping_serialization_path, "w") as text_file:
                        json.dump(mapping_dict, text_file)
                except Exception as e:
                    exception_message = ' with exception: \'{}\'' % e
                    eprint(exception_message)
            should_run_similarity_sanity_check = False
            if should_run_similarity_sanity_check:
                for bbox in character_detections.CharacterBoundingBoxes:
                    if_exist = [m for m in mapping if m.ThumbnailId == bbox.ThumbnailId and m.BoxesConsolidation < 0]
                    if len(if_exist) == 0:
                        continue

                    ids.append(bbox.ThumbnailId)
                    features.append(bbox.Features)
                sanity_check_num_examples = 100
                similar_pairs, dissimilar_pairs = find_similar_and_dissimilar_pairs(sanity_check_num_examples, ids,
                                                                                    features)
                sim_repo = os.path.join(role_path, "Similar")
                visualize_similarity_features(sim_repo, similar_pairs, role_detections_repo)

                dis_repo = os.path.join(role_path, "DisSimilar")
                visualize_similarity_features(dis_repo, dissimilar_pairs, role_detections_repo)

            # copy all bboxes grouped by cluster id
            groups_root = os.path.join(role_path, '', 'groups')
            if os.path.isdir(groups_root):
                shutil.rmtree(groups_root)
                time.sleep(2)
            os.mkdir(groups_root)

            noise_repo = os.path.join(groups_root, '', 'All_noisy_clusters')
            for bbox in mapping:
                cluster_repo = os.path.join(groups_root, '', 'Cluster_{}'.format(bbox.BoxesConsolidation))
                if bbox.BoxesConsolidation < 0:
                    cluster_repo = noise_repo

                if not os.path.isdir(cluster_repo):
                    os.mkdir(cluster_repo)

                bbox_target = os.path.join(cluster_repo, '', '{}.jpg'.format(bbox.ThumbnailId))
                if not os.path.isfile(bbox_target):
                    source_bbox = os.path.join(role_detections_repo, '', '{}.jpg'.format(bbox.ThumbnailId))
                    shutil.copyfile(source_bbox, bbox_target)

            # make collages
            collage_repo = os.path.join(groups_root, '', 'All_collages')
            if os.path.isdir(collage_repo):
                shutil.rmtree(collage_repo)
            os.mkdir(collage_repo)
            for cluster_repo_name in os.listdir(groups_root):
                if not cluster_repo_name.startswith('Cluster_'):
                    continue

                cluster_repo_path = os.path.join(groups_root, '', cluster_repo_name)
                collage_images = [os.path.join(cluster_repo_path, '', bbox_name) for bbox_name in os.listdir(cluster_repo_path)]
                target_collage_path = os.path.join(collage_repo, '', '{}.jpg'.format(cluster_repo_name))
                create_collage(collage_images, target_collage_path)

            # copy negative examples
            neg_dir = os.path.join(groups_root, '', 'negatives')
            if not os.path.isdir(neg_dir):
                os.mkdir(neg_dir)
            negative_examples = DetectionMapping.parse_negatives(grouping_output_path)
            ordered_negs = sorted(negative_examples, key=lambda neg: neg['BoundingBox']['Width']*neg['BoundingBox']['Height'], reverse=True)
            num_negs = min(300, len(ordered_negs))
            top_negs = ordered_negs[0:num_negs]
            keyframes_dir = os.path.join(role_path, '', '_KeyFrameThumbnail')
            for top_neg in top_negs:
                keyframe_thumbnail_id = top_neg['KeyframeId']
                keyframe_thumbnail_path = os.path.join(keyframes_dir, '', 'KeyFrameThumbnail_{}.jpg'.format(top_neg['KeyframeId']))
                x = top_neg['BoundingBox']['X']
                y = top_neg['BoundingBox']['Y']
                w = top_neg['BoundingBox']['Width']
                h = top_neg['BoundingBox']['Height']
                neg_image_target_path = os.path.join(neg_dir, '', '{}_{}-{}-{}-{}.jpg'
                                                     .format(keyframe_thumbnail_id, x, y, w, h))
                crop = crop_image(keyframe_thumbnail_path, x, y, w, h)
                save_image(neg_image_target_path, crop)

    return


if __name__ == "__main__":
    main()
