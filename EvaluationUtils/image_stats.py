from PIL import Image
import numpy as np
import pandas as pd
from Animator.bbox_grouper_api import CharacterDetectionOutput
from sklearn.metrics import confusion_matrix
from confusion_matrisizer import plot_cm
import os
from sklearn import metrics
import shutil


def get_image_size(image_path):
    # get width and height
    with Image.open(image_path) as pil:
        width, height = pil.size
        return width, height


def draw_confusion_matrix(pre_cls, gt_cls, tag_list, series_name, root_output_dir, log_output_file_name):
    mtx = confusion_matrix(gt_cls, pre_cls, labels=tag_list)
    print(mtx)
    log_path = os.path.join(root_output_dir, log_output_file_name)
    with open(log_path, mode='a') as cm:
        cm.write('Ordered_classes:\n{}\n'.format(series_name))
        cm.write(str(mtx))
        cm.write('\n')
        report = metrics.classification_report(gt_cls, pre_cls, digits=4)
        print(report)
        cm.write(str(report))
    cm_fig_path = os.path.join(root_output_dir, 'FullEpisode_FilteredLowRes_MyCM-{}.png'.format(series_name))
    plot_cm(mtx, tag_list, normalize=False, title='Confusion Matrix {}'.format(series_name), save_figure_path=cm_fig_path)
    return mtx


def filter_insignificant_images_and_run_cm():
    root_dir = r'???'
    predictions_and_gt_df_path = root_dir + r"\FullEpisodeTest\Labels\Output\The Land Before Time\Evaluation_The Land Before Time.tsv"
    animation_detection_output_path = root_dir + r"\SeResNext\Videos\The Land Before Time\Test\animationdetectionoutput.json"
    tag_list = ['Unknown', 'Ducky', 'Cera', 'LittleFoot', 'Petrie', 'Spike', 'Chomper', 'Ruby', 'DaddyTopps']
    series_name = 'The Land Before Time'
    root_output_dir = root_dir + r'\FullEpisodeTest\Labels\Output\The Land Before Time\FilteredSizeAndConf'
    if os.path.isdir(root_output_dir):
        shutil.rmtree(root_output_dir)
    os.mkdir(root_output_dir)

    log_output_file_name = 'logFile-The Land Before Time-FilteredSizeAndConf.txt'
    pred_gt_df = pd.read_csv(predictions_and_gt_df_path, sep='\t')
    images = pred_gt_df['Id']
    widths = list()
    heights = list()
    areas = list()
    confidencs = []
    gts = pred_gt_df['MappedLabel'].values

    # get detection confidence
    detections = CharacterDetectionOutput.read_from_json(animation_detection_output_path)
    guid_to_detection_conf = {cd.ThumbnailId: cd.Confidence for cd in detections.CharacterBoundingBoxes}

    for i in range(len(images)):
        im = images[i]
        w, h = get_image_size(im)
        widths.append(w)
        heights.append(h)
        areas.append(w*h)
        image_guid = im.split('\\')[-1].split('.')[0]

        confidence = guid_to_detection_conf.get(image_guid, 0)
        confidencs.append(confidence)

    pred_gt_df['Width'] = pd.Series.from_array(np.asarray(widths))
    pred_gt_df['Height'] = pd.Series.from_array(np.asarray(heights))
    pred_gt_df['Area'] = pd.Series.from_array(np.asarray(areas))
    pred_gt_df['DetectionConf'] = pd.Series.from_array(np.asarray(confidencs))
    pred_gt_df['GT'] = pd.Series.from_array(np.asarray(gts))

    # filter out bad input
    min_confidence = 0.2
    min_area = 3500
    min_width = 50
    min_height = 50
    print('1. Initial N={}'.format(pred_gt_df.shape[0]))
    pred_gt_df = pred_gt_df[pred_gt_df.DetectionConf >= min_confidence]
    print('2. Detect  N={}'.format(pred_gt_df.shape[0]))

    pred_gt_df = pred_gt_df[pred_gt_df.Width >= min_width]
    print('3. Width   N={}'.format(pred_gt_df.shape[0]))

    pred_gt_df = pred_gt_df[pred_gt_df.Height >= min_height]
    print('4. Height  N={}'.format(pred_gt_df.shape[0]))

    pred_gt_df = pred_gt_df[pred_gt_df.Height * pred_gt_df.Width >= min_area]
    print('5. Area    N={}'.format(pred_gt_df.shape[0]))

    filtered_df_output_path = os.path.join(root_output_dir, 'filterd_df.tsv')
    pred_gt_df.to_csv(filtered_df_output_path, sep='\t')

    # re-plot the confusion matrix
    predictions = pred_gt_df.Prediction
    gt = pred_gt_df.GT
    cm = draw_confusion_matrix(predictions, gt, tag_list, series_name, root_output_dir, log_output_file_name)
    return


def copy_files(file_paths, target_repo_path, file_names=None):
    if os.path.isdir(target_repo_path):
        shutil.rmtree(target_repo_path)
    os.mkdir(target_repo_path)

    for i in range(len(file_paths)):
        source_file_path = file_paths.iloc[i]
        file_name = '{}_{}'.format(file_names.iloc[i], os.path.basename(source_file_path))
        target_file_path = os.path.join(target_repo_path, file_name)
        shutil.copyfile(source_file_path, target_file_path)


def error_analysis():
    root_dir = r'???'
    pred_gt_log_path = root_dir + r"\FullEpisodeTest\Labels\Output\The Land Before Time\Evaluation_The Land Before Time.tsv"
    root_path = root_dir + r"\FullEpisodeTest\Labels\Output\The Land Before Time"
    df = pd.read_csv(pred_gt_log_path, sep='\t')
    false_positives = df[df['MappedLabel'].eq('Unknown') & ~df['Prediction'].eq('Unknown')]
    false_negatives = df[~df['MappedLabel'].eq('Unknown') & df['Prediction'].eq('Unknown')]

    fn_repo = os.path.join(root_path, 'FN')
    fp_repo = os.path.join(root_path, 'FP')

    fn_paths = false_negatives['Id']
    fn_file_names = false_negatives['MappedLabel']
    fp_paths = false_positives['Id']
    fp_file_names = false_positives['Prediction']

    copy_files(fn_paths, fn_repo, fn_file_names)
    copy_files(fp_paths, fp_repo, fp_file_names)


def collect_small_insignificant_images(character_repos, small_images_repo, animation_detection_output_json):
    min_side = 70
    min_conf = .5
    min_area = 5000
    detections_output = CharacterDetectionOutput.read_from_json(animation_detection_output_json)
    guid_to_detect_conf = {cd.ThumbnailId: cd.Confidence for cd in detections_output.CharacterBoundingBoxes}

    for character_repo in character_repos:
        character_images = [os.path.join(character_repo, cim) for cim in os.listdir(character_repo)]
        for character_image in character_images:
            width, height = get_image_size(character_image)
            im_id = os.path.basename(character_image).split('.')[0]
            confidence = guid_to_detect_conf.get(im_id, 0)

            # insignificant to side repo
            if width < min_side or height < min_side or confidence < min_conf or width*height < min_area:
                target_path = os.path.join(small_images_repo, os.path.basename(character_image))
                shutil.copyfile(character_image, target_path)
                os.remove(character_image)


def move_insignificant_detections():
    root_dir = "???"
    video_root = root_dir + r'\FullEpisodeTest\Labels\Evaluation\The Land Before Time'
    detect_output = root_dir + r"\SeResNext\Videos\The Land Before Time\Test\animationdetectionoutput.json"
    characters_repos = [os.path.join(video_root, cr) for cr in os.listdir(video_root) if 'small' not in cr]
    small_repo = os.path.join(video_root, 'small')
    collect_small_insignificant_images(characters_repos, small_repo, detect_output)


if __name__ == '__main__':
    filter_insignificant_images_and_run_cm()
    print('Done!')
