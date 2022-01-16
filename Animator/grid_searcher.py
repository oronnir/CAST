import sys
import os
import shutil
import time
import json
import pandas as pd
import Animator.utils
from . import batch_grouper
from . import clustering_methods as cm
from . import normalization_methods as nm


DATA_DIR = sys.argv[1]
GRID_SEARCH_DIR = sys.argv[2]


def join_labels_features_data_frame(categories_with_labels, categories_with_features):
    joined_data_frame = pd.merge(categories_with_labels, categories_with_features, on='Category', how='inner')
    return joined_data_frame


def train_model(input_dir, hyper_params_permutation, iteration_n):
    print('starting iteration #' + str(iteration_n))

    # create output repository
    iteration_dir_path = os.path.join(GRID_SEARCH_DIR, "", "iter_" + str(iteration_n))
    os.mkdir(iteration_dir_path)

    # run full batch
    batch_stats = batch_extractor.group_bboxes_batch(input_dir, iteration_dir_path, hyper_params_permutation)

    # collect batch stats
    batch_stats['hyper_params'] = json.dumps({k: v.__name__ if callable(v) else v
                                              for k, v in hyper_params_permutation.items()})
    return batch_stats


def main():
    """run full grid search using hyper params and train_model()"""
    print("Starting grid search")
    start_time = time.time()
    params = dict(
        clustering_method=[
            cm.modularity_maximization_06_coframe_penalty_045_time995,
            cm.agglomerative_at_k,
            cm.k_means,
            cm.spectral,
            cm.mean_shift,
            cm.agglomerative_at_k_complete,
        ],
        normalization_method=[
            nm.identity,
            nm.l2,
            nm.pca_whitening_30d
        ],
        min_confidence_percentile=['20'],
        min_short_edge_absolute=['20'],
        min_short_edge_ratio=['0.025'],
        max_short_edge_absolute=['1000'],
        max_short_edge_ratio=['1.0'],
        min_cluster_significance=['0.55'],
        keep_cluster_percentile=['80']
    )

    # create a new folder for grid search if not exist
    if not os.path.isdir(GRID_SEARCH_DIR):
        os.mkdir(GRID_SEARCH_DIR)

    # remove old results
    previous_run_folders = os.listdir(GRID_SEARCH_DIR)
    if len(previous_run_folders) > 0:
        for old_dir in previous_run_folders:
            dir_path = os.path.join(GRID_SEARCH_DIR, "", old_dir)
            if os.path.isfile(dir_path):
                os.remove(dir_path)
            else:
                folder_to_remove = dir_path
                shutil.rmtree(folder_to_remove)

    # run grid search
    all_iterations_stats = os.path.join(GRID_SEARCH_DIR, '', 'all_iterations_stats.tsv')
    results = Animator.vi_utils.grid_search(DATA_DIR, train_model, params, all_iterations_stats)
    print(results)
    print("End grid search run after: {:.3f}sec".format(time.time() - start_time))
    print('The aggregated results are available at: {}'.format(all_iterations_stats))


if __name__ == '__main__':
    main()
