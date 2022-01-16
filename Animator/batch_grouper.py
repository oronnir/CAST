import os
import sys
import json
import time
import shutil
import numpy as np
import pandas as pd
from termcolor import colored
from Animator.utils import eprint, profiling, hash_values
import Animator.normalization_methods as nm
import Animator.clustering_methods as cm
import Animator.consolidator as bbg
from Animator.cluster_logic import GroupingEvaluator


# read configuration
localProjectDir = ''
with open(os.path.join(os.path.dirname(__file__), '', 'config.json')) as data_file:
    data_file_objects = json.load(data_file)
    locals().update(data_file_objects)

MANGA109 = True
book_name_to_cropped_detections_folder = lambda fold: f'\\..\\Manga109\\Manga109_released_2020_09_26\\images\\{fold}\\Detections\\animationdetectionoriginalimages'
EVALUATION_DS = False
VIACOM = False
YOUTUBE_DS = False

RUN_VISUALIZATION_ONLY = False
EVALUATION_FILE_TAG = 'EVAL.tsv'
VISUALIZATION_FOLDER_NAME = 'visualizations'
root_youtube_dataset_folder = r'\..\Grouping\DebugVideos' if RUN_VISUALIZATION_ONLY else r'???\YouTubeEval\SeResNext'
youtube_foreign_keys_bid_init_to_shortid = dict([
    ('?', '?'),
])
youtube_foreign_keys_shortid_to_bid_init = dict(zip(youtube_foreign_keys_bid_init_to_shortid.values(),
                                                    youtube_foreign_keys_bid_init_to_shortid.keys()))


def create_clustering_visualizations(clustering_visualization_folder_path, bboxes_groups, source_thumbnails_folder,
                                     detected_bboxes):
    os.mkdir(clustering_visualization_folder_path)
    bbox_id_to_name = dict([(box.Id, box.ThumbnailId) for box in detected_bboxes.CharacterBoundingBoxes]) \
        if RUN_VISUALIZATION_ONLY \
        else dict([(box.Id, box.KeyFrameIndex) for box in detected_bboxes.CharacterBoundingBoxes])
    at_least_one_thumb_exist_flag = False

    for bbox_group in bboxes_groups:
        cluster_folder_path = os.path.join(clustering_visualization_folder_path, '',
                                           "{}_Cluster_{}".format('Noise' if bbox_group.ClusterId < 0 else 'Good', bbox_group.ClusterId))
        if not os.path.isdir(cluster_folder_path):
            os.mkdir(cluster_folder_path)
        source_thumbnail_name = f'{bbox_group.ThumbnailId}.jpg' if MANGA109 else f'thumbnail_{bbox_id_to_name[bbox_group.Id]}'
        source_thumbnail_path = os.path.join(source_thumbnails_folder, '', source_thumbnail_name)
        destination_thumbnail_path = os.path.join(cluster_folder_path, '', '{}{}.jpg'
                                                  .format(bbox_id_to_name[bbox_group.Id], '_BEST'
                                                                                          if bbox_group.IsBest else ''))
        if os.path.isfile(source_thumbnail_path):
            at_least_one_thumb_exist_flag = True
            shutil.copy(source_thumbnail_path, destination_thumbnail_path)

    print('%%%%%%%%%%%%%%%%%' + ('Found at least 1 thumb' if at_least_one_thumb_exist_flag else 'NO BBOXES!!!!') +
          '%%%%%%%%%%%%%%%%%')


def validate_best_face(bboxes_groups):
    cluster_ids = set([cid.ClusterId for cid in bboxes_groups])
    for cid in cluster_ids:
        if cid < 0:
            continue
        cluster_group_best_thumbnail = [box for box in bboxes_groups if box.ClusterId == cid and box.IsBest]
        best_thumbnails_in_group = len(cluster_group_best_thumbnail)
        if best_thumbnails_in_group != 1:
            continue


def create_clusters_visualization(video_output_folder, video_name, bboxes_groups, detected_bboxes):
    if YOUTUBE_DS:
        clustering_visualization_folder_path = os.path.join(video_output_folder, VISUALIZATION_FOLDER_NAME)
        old_bid_init_to_bid = dict([(bid.split('-')[0], bid) for bid in os.listdir(root_youtube_dataset_folder)])
        source_thumbnails_folder_name = old_bid_init_to_bid[video_name.split('_')[0]]
        source_thumbnails_folder = os.path.join(root_youtube_dataset_folder, '', source_thumbnails_folder_name)
        create_clustering_visualizations(clustering_visualization_folder_path, bboxes_groups, source_thumbnails_folder,
                                         detected_bboxes)
    elif VIACOM:
        raise NotImplementedError()
    elif EVALUATION_DS:
        raise NotImplementedError()
    elif MANGA109:
        clustering_visualization_folder_path = os.path.join(video_output_folder, VISUALIZATION_FOLDER_NAME)
        source_thumbnails_folder = book_name_to_cropped_detections_folder(video_name)
        create_clustering_visualizations(clustering_visualization_folder_path, bboxes_groups, source_thumbnails_folder,
                                         detected_bboxes)
    else:
        raise NotImplementedError()
    return


def group_bboxes_batch(input_folder, output_path, hyper_parameters):
    # get hyper parameters
    clustering_function = hyper_parameters['clustering_method']
    normalization_method = hyper_parameters['normalization_method']
    min_confidence_percentile = hyper_parameters['min_confidence_percentile']
    min_short_edge_absolute = hyper_parameters['min_short_edge_absolute']
    min_short_edge_ratio = hyper_parameters['min_short_edge_ratio']
    max_short_edge_absolute = hyper_parameters['max_short_edge_absolute']
    max_short_edge_ratio = hyper_parameters['max_short_edge_ratio']
    min_cluster_significance = hyper_parameters['min_cluster_significance']
    keep_cluster_percentile = hyper_parameters['keep_cluster_percentile']

    # set i/o
    local_videos_dir = os.path.join(localProjectDir, "", input_folder)
    videos = [os.path.join(local_videos_dir, '', p) for p in os.listdir(local_videos_dir)]
    num_videos = len(videos)
    evaluation_acc_path = os.path.join(output_path, '', 'batch_eval_stat.tsv')
    batch_evaluations = pd.DataFrame()

    for i in range(num_videos):
        time_marker = time.time()
        time_marker = profiling("Parsed args", time_marker)
        video = videos[i]
        video_name = os.path.basename(video).replace(".json", '')
        video_output_folder = os.path.join(output_path, "", video_name)

        # read labels
        input_file_path = os.path.join(video, 'animationdetectionoutput.json')
        output_file = os.path.join(video_output_folder, "", 'output_file.json')
        video_args = (None, '--input', input_file_path, '--output', output_file,
                      "--min-confidence-percentile", min_confidence_percentile,
                      "--min-short-edge-absolute", min_short_edge_absolute,
                      "--min-short-edge-ratio", min_short_edge_ratio,
                      "--max-short-edge-absolute", max_short_edge_absolute,
                      "--max-short-edge-ratio", max_short_edge_ratio,
                      "--min-cluster-significance", min_cluster_significance,
                      "--keep-cluster-percentile", keep_cluster_percentile
                      )

        bbox_grouper = bbg.BboxGrouper(args=video_args, cluster_analysis=clustering_function,
                                       normalization_method=normalization_method)
        bbox_grouper_parsed_args = bbg.read_and_validate_args(video_args)
        hyper_params = dict(normalization_method=normalization_method.__name__,
                            cluster_analysis=clustering_function.__name__)
        detected_bboxes = bbox_grouper_parsed_args[0]
        grouping_evaluator = GroupingEvaluator(hyper_params)

        print(colored("Start analyzing file #{} out of {} named: {}".format(i + 1, num_videos,
                                                                            os.path.basename(video_name)), 'yellow'))

        # extract features and group single video
        bboxes_groups = bbox_grouper.group_characters_single_video(time_marker)

        # copy thumbnails per cluster for visualization
        create_clusters_visualization(video_output_folder, video_name, bboxes_groups, detected_bboxes)

        # validate a single best thumbnail per cluster
        validate_best_face(bboxes_groups)

        # filter the unlabeled bounding boxes
        labeled_thumbnail_ids = {detected_bbox.ThumbnailId for detected_bbox in detected_bboxes.CharacterBoundingBoxes if detected_bbox.IsLabeled}
        features = np.asarray([detected_bbox.Features for detected_bbox in detected_bboxes.CharacterBoundingBoxes if detected_bbox.IsLabeled])
        gt_labels_names = np.asarray(
            [detected_bbox.Character for detected_bbox in detected_bboxes.CharacterBoundingBoxes if detected_bbox.IsLabeled])
        gt_labels_ids = np.asarray(hash_values(gt_labels_names))

        # evaluate the current file
        predicted_labels = [bbox_group.ClusterId for bbox_group in bboxes_groups if bbox_group.ThumbnailId in labeled_thumbnail_ids]
        current_video_evaluation = evaluate_single_file(grouping_evaluator, features, predicted_labels, gt_labels_ids,
                                                        video_name, normalization_method, clustering_function)

        # accumulate all evaluations in batch_evaluations as a pd.DF
        evaluation_file_path = os.path.join(video_output_folder, "", 'output_file_{}'.format(EVALUATION_FILE_TAG))

        current_video_evaluation.to_csv(evaluation_file_path, sep='\t')
        batch_evaluations = batch_evaluations.append(current_video_evaluation, ignore_index=True)

        if not os.path.isfile(evaluation_file_path):
            raise AttributeError("Error: eval file is not available at: " + evaluation_file_path)

    # save the statistics from the evaluations dataframe into the results folder
    batch_evaluations.to_csv(evaluation_acc_path, sep='\t')
    aggregated_batch_stats = batch_evaluations.filter(['noise_count', 'p_noise', 'homogenity', 'completeness',
                                                       'v_measure', 'random_index', 'adjusted_mutual_info',
                                                       'normalized_mutual_info', 'silhouette', 'itzi_k']) \
        .aggregate(['min', 'max', 'mean', 'median', 'std', 'count'])
    print('batch  finished\n{}'.format(aggregated_batch_stats))
    aggregated_batch_stats.to_csv(evaluation_acc_path.replace('.tsv', '_aggregation.tsv'), sep='\t')
    return aggregated_batch_stats


def evaluate_single_file(grouping_evaluator, features, predicted_labels, gt_labels_ids, video_name,
                         normalization_method, clustering_function):
    episode_clustering_evaluation = grouping_evaluator.evaluate_clusters(features, predicted_labels, gt_labels_ids,
                                                                         video_name)
    episode_clustering_evaluation['normalization_method'] = normalization_method.__name__
    episode_clustering_evaluation['cluster_analysis'] = clustering_function.__name__
    del episode_clustering_evaluation['hyper_params']
    current_video_evaluation = pd.DataFrame.from_dict({k: [v] for k, v in episode_clustering_evaluation.items()})
    return current_video_evaluation


def append_all_csvs(directory, file_name):
    directory = str.replace(directory, '\\', '/')
    batch_file_full_path = directory + '/' + file_name
    if os.path.isfile(batch_file_full_path):
        os.remove(batch_file_full_path)

    csvs = [directory + '/' + f for f in os.listdir(directory) if f.__contains__('.csv')]
    first_flag = True
    with open(batch_file_full_path, 'wb') as fd:
        for csv in csvs:
            try:
                df = pd.read_csv(csv)
                df.to_csv(fd, header=first_flag)
            except Exception as e:
                eprint("exception in append csvs on file: " + csv + '\nError message:\n' + str(e), e)
                continue
            first_flag = False
    return


def read_and_validate_args(args):
    num_of_args = 3
    args_expected = "<input_folder> <output_file_path>"
    if args is None or len(args) is not num_of_args:
        eprint(
            "invalid input - missing arguments. Expected: \"" + args_expected + "\" ... And instead got " + str(args))
        raise ValueError('Invalid args input.')
    input_folder = args[1]
    output_file_path = args[2]

    if not os.path.isdir(input_folder) or len(os.listdir(input_folder)) == 0 or os.path.isfile(input_folder):
        eprint("Invalid input file")
        raise ValueError('Invalid args input.')

    output_file_containing_folder = os.path.abspath(os.path.join(output_file_path, os.pardir))
    if not os.path.isdir(output_file_containing_folder):
        eprint('output file\'s folder is invalid.')
        raise ValueError('Invalid args input.')

    return input_folder, output_file_path


def main():
    print("\nStart batch analysis")
    input_folder, output_file_path = read_and_validate_args(sys.argv)

    # clear previous run outputs
    if os.path.isfile(output_file_path):
        os.remove(output_file_path)

    # clear the output folder
    output_folder = os.path.abspath(os.path.join(output_file_path, os.pardir))
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder, ignore_errors=True, onerror=None)
    os.mkdir(output_folder)

    # group all videos in batch
    hyper_params = dict(normalization_method=nm.identity,
                        clustering_method=cm.k_means,
                        min_confidence_percentile=str(0),
                        min_short_edge_absolute=str(0),
                        min_short_edge_ratio=str(0),
                        max_short_edge_absolute=str(10000),
                        max_short_edge_ratio=str(1),
                        min_cluster_significance=str(0),
                        keep_cluster_percentile=str(0))

    group_bboxes_batch(input_folder, output_folder, hyper_params)
    print("End batch analysis")
    return


if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        eprint('Failed with exception {}'.format(exception), exception)
