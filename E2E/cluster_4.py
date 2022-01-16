import os
import sys
import pandas as pd

from EvaluationUtils.post_processor import post_process_single_episode
from Animator.consolidator import bbox_grouper_main


def cluster_main():
    dataset_path = r'???\SAIL_animation_movie_character_database\finetuned'
    stats_df_path = os.path.join(dataset_path, r'ClusteringStats.tsv')
    predictions_df = pd.DataFrame({'SeriesName': [], 'Role': [], 'NumProposals_1': [], 'InitialK_2': [],
                                   'DbscanK_3': [], 'ReClusterK_4': [], 'DicardedK_5': [], 'FinalK_6': [],
                                   'AvgNumProposalsPerCluster': [], 'ValidBoxes': []})
    # fresh start
    if os.path.isfile(stats_df_path):
        os.remove(stats_df_path)

    series = [s for s in os.listdir(dataset_path)]
    for ser in series:
        for role in ['Training']:
            print('$$$ start analyzing {} $$$'.format(ser))
            input_path = os.path.join(dataset_path, ser, role, 'animationdetectionoutput.json')
            output_path = os.path.join(dataset_path, ser, role, 'animationgroupingoutput.json')

            # cleanup
            if os.path.isfile(output_path):
                os.remove(output_path)

            # run grouper
            sys.argv = [__file__, '-i', input_path, '-o', output_path]
            bbox_grouper_main()

            # run post processor
            pred_row = post_process_single_episode(dataset_path, ser, role)

            predictions_df = predictions_df.append(pred_row, ignore_index=True)

            print('Finished an episode...')
        predictions_df.to_csv(stats_df_path, header=True, sep='\t')


if __name__ == '__main__':
    cluster_main()
    print('DONE!')
