import glob
import time
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from msrest.authentication import ApiKeyCredentials
from sklearn import metrics
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
from Animator.utils import eprint, recreate_dir
from EvaluationUtils.confusion_matrisizer import plot_cm
from sklearn.utils.extmath import softmax


# triplets
ser_to_project_id = {
    'BobTheBuilder': '',
    'FairlyOddParents': '',
    'FiremanSam': '',
    'Floogals': '',
    'Garfield': '',
    'Southpark': '',
    'The Land Before Time': ''
}

ser_to_iter = {
    'BobTheBuilder': 'Iteration1',
    'FairlyOddParents': 'Iteration1',
    'FiremanSam': 'Iteration1',
    'Floogals': 'Iteration1',
    'Garfield': 'Iteration1',
    'Southpark': 'Iteration1',
    'The Land Before Time': 'Iteration1'
}

ENDPOINT = "https://???-prediction.cognitiveservices.azure.com/"
PREDICTION_KEY = ''
PREDICTION_ID = "/subscriptions/???/.../???/providers/Microsoft.CognitiveServices/accounts/???-Prediction"
UNKNOWN_TH = -1  # set to 0.5 if you want Unknown


class ClassificationPredictor(object):
    def __init__(self, output_path):
        print("Initializing with prediction API of customvision.ai")
        prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
        self.prediction_api = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)
        self.all_gt, self.all_pred = np.array([]), np.array([])
        self.tag_dict = dict(Unknown=0) if UNKNOWN_TH > 0 else dict()
        self.DataFilePath = output_path
        if os.path.isfile(self.DataFilePath):
            os.remove(self.DataFilePath)

    def predict_image_dictionary(self, project_id, iteration_name, testset_path, is_labeled=True):
        if not os.path.exists(testset_path):
            print("file path {0} not exist".format(testset_path))
            return

        all_class = glob.glob(os.path.join(testset_path, '*')) if is_labeled else [testset_path]
        test_set_labels = set(os.path.basename(f) for f in all_class) if is_labeled else {
            os.path.basename(testset_path)}
        intersection_classes = None
        idx_to_mapped_label = None

        print("test images for dataset {0}".format(testset_path))

        predictions_df = pd.DataFrame({'Id': [], 'MappedLabel': [], 'OriginalLabel': [], 'Prediction': [],
                                       'OriginalPrediction': [], 'Probability': [], 'RawResult': []})

        for class_path in all_class:
            # delete the Thumbs.db file
            img_paths = list(filter(lambda x: not x.endswith(".db"), glob.glob(os.path.join(class_path, '*'))))
            idx_name = os.path.basename(class_path)

            if not img_paths:
                print('No images for class: {}!'.format(idx_name))
                continue

            print("test images for category {0} from {1}".format(idx_name, class_path))
            for img_path in img_paths:
                with open(img_path, 'rb') as test_data:
                    results = None
                    try:
                        time.sleep(1)
                        results = self.prediction_api.classify_image_with_no_store(project_id=project_id,
                                                                                   image_data=test_data,
                                                                                   published_name=iteration_name)
                    except Exception as exception:
                        eprint('Exception thrown at image: {}'.format(img_path), exception)
                        continue

                    # add exclusive test set labels
                    has_unknown = 1 if UNKNOWN_TH > 0 else 0
                    if not intersection_classes:
                        model_classes = {p.tag_name.split('_')[0] for p in results.predictions}
                        if UNKNOWN_TH < 0:
                            model_classes -= {'Negative', 'Unknown'}
                        model_labels = {c for c in model_classes}
                        intersection_classes = test_set_labels & model_labels if is_labeled else model_labels
                        current_id = has_unknown
                        for model_class in model_classes:
                            if model_class in intersection_classes:
                                self.tag_dict[model_class] = current_id
                                current_id += 1
                            else:
                                self.tag_dict[model_class] = 0

                        if is_labeled:
                            # classes in test episode but not in training/model
                            additional_classes = list(test_set_labels - {c for c in model_classes})
                            for additional_class in additional_classes:
                                self.tag_dict[additional_class] = 0
                            idx_to_mapped_label = {t[1]: t[0] for t in self.tag_dict.items()
                                                   if t[0] not in ['Unknown', 'Negative']}# or t[1] != 0}

                    # add training labels

                    num_cls = has_unknown + (len(intersection_classes) if is_labeled else len(model_classes))
                    if self.all_pred.size == 0:
                        self.all_pred = np.zeros([0, num_cls])
                    if self.all_gt.size == 0:
                        self.all_gt = np.zeros([0, num_cls], dtype=np.int8)
                    _gt, _pred = np.zeros([1, num_cls], dtype=np.int8), np.zeros([1, num_cls])
                    for pre in results.predictions:
                        label_name = pre.tag_name.split('_')[0]
                        if UNKNOWN_TH < 0 and label_name in {'Unknown', 'Negative'}:
                            continue
                        label_index = self.tag_dict[label_name]
                        _pred[0, label_index] = max(_pred[0, label_index], pre.probability)

                    # final_prediction
                    best_prediction = sorted(results.predictions, key=lambda t: t.probability, reverse=True)[0]
                    final_prediction_tag = best_prediction.tag_name.split('_')[0]
                    final_prediction_probability = best_prediction.probability
                    if final_prediction_probability < UNKNOWN_TH:
                        final_prediction_tag = 'Unknown'
                        final_prediction_probability = 1.0 - final_prediction_probability
                        _pred[0, 0] = final_prediction_probability
                    elif final_prediction_tag not in intersection_classes and UNKNOWN_TH > 0:
                        final_prediction_tag = 'Unknown'
                        final_prediction_probability = 1.0
                        _pred[0, 0] = final_prediction_probability

                    # _pred = softmax(10.0*_pred)
                    self.all_pred = np.vstack((self.all_pred, _pred))
                    if is_labeled:
                        _gt[0, self.tag_dict[idx_name]] = 1
                        self.all_gt = np.vstack((self.all_gt, _gt))

                    # keep predictions in a dataframe
                    raw_preds = "[{}]".format(','.join('{}:{}'.format(p.tag_name, p.probability)
                                                       for p in results.predictions))
                    if idx_name not in self.tag_dict or self.tag_dict[idx_name] not in idx_to_mapped_label:
                        stop = 0
                    mapped_label = idx_to_mapped_label[self.tag_dict[idx_name]]
                    pred_row = dict(Id=img_path,
                                    MappedLabel=mapped_label if is_labeled else None,
                                    OriginalLabel=idx_name,
                                    Prediction=final_prediction_tag,
                                    OriginalPrediction=best_prediction.tag_name,
                                    Probability=final_prediction_probability,
                                    RawResult=raw_preds)
                    predictions_df = predictions_df.append(pred_row, ignore_index=True)
            print("pred for class {0} is done".format(idx_name))

        print("test succeed for dataset {0}".format(testset_path))
        predictions_df.to_csv(self.DataFilePath, header=True, sep='\t')
        if is_labeled:
            print(metrics.classification_report(predictions_df.MappedLabel, predictions_df.Prediction, digits=4))


class ClassificationEvaluator(object):
    def __init__(self, evaluation_root):
        self.Root = evaluation_root
        self.SeriesName = os.path.basename(evaluation_root)
        ap_image_name = 'AveragePrecisionCurve-{}.png'.format(self.SeriesName)
        self.AveragePrecisionCurvePath = os.path.join(self.Root, ap_image_name)

        if os.path.isfile(self.AveragePrecisionCurvePath):
            os.remove(self.AveragePrecisionCurvePath)

        self.ConfusionMatLogPath = os.path.join(self.Root, 'ConfusionMatLog-{}.txt'.format(self.SeriesName))
        if os.path.isfile(self.ConfusionMatLogPath):
            os.remove(self.ConfusionMatLogPath)

    def draw_curve_pre_cls(self, v_pred, v_gt, cls_name):
        plt.figure(figsize=(14, 12))
        precision, recall, _ = precision_recall_curve(v_gt, v_pred)
        average_precision = average_precision_score(v_gt, v_pred)
        print("average_precision: {}".format(average_precision))
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('{0} Precision-Recall curve: AP={1:0.4f}'.format(cls_name, average_precision))
        fig_ap = plt.gcf()
        fig_ap.savefig(self.AveragePrecisionCurvePath, format='png', dpi=100)
        plt.show(block=False)
        time.sleep(4)
        plt.close('all')
        return

    def draw_confusion_matrix(self, predicted_vectors, ground_truth_vectors, tag_dict):
        ordered_tuples = sorted(tag_dict.items(), key=lambda tup: tup[1])
        mapped_label_tuples = [t for t in ordered_tuples if (t[0] is 'Unknown' or t[1] != 0) or UNKNOWN_TH < 0]

        print('Confusion matrix for classes: {}'.format(ordered_tuples))
        mapped_label_names = [t[0] for t in mapped_label_tuples]
        index_dict = {v: k for k, v in mapped_label_tuples}
        gt_label_indices = ground_truth_vectors.argmax(axis=1)
        prediction_scores = predicted_vectors.argmax(axis=1)

        gt_cls = [index_dict[cls_idx] for cls_idx in gt_label_indices]
        pre_cls = [index_dict[cls_idx] for cls_idx in prediction_scores]
        mtx = confusion_matrix(gt_cls, pre_cls, labels=mapped_label_names)
        print(mtx)
        with open(self.ConfusionMatLogPath, mode='a') as cm:
            cm.write('Ordered_classes:\n{}\n'.format(ordered_tuples))
            cm.write(str(mtx))
            cm.write('\n')
            report = metrics.classification_report(gt_cls, pre_cls, digits=4)
            cm.write(str(report))
        cm_fig_path = os.path.join(self.Root, 'MyCM-{}.png'.format(self.SeriesName))
        plot_cm(mtx, mapped_label_names, normalize=True,
                title='Confusion matrix {}'.format(self.SeriesName), save_figure_path=cm_fig_path)
        return mtx


def pred_classification_project(project_id, iteration_name, test_dictionary, series_eval_repo, is_labeled=True):
    series_name = os.path.basename(series_eval_repo)
    predictions_output_csv = os.path.join(series_eval_repo, 'Evaluation_{}.tsv'.format(series_name))
    predictor = ClassificationPredictor(predictions_output_csv)
    predictor.predict_image_dictionary(project_id, iteration_name, test_dictionary, is_labeled)
    confusion_mat = None

    # plot evaluations
    if is_labeled:
        evaluator = ClassificationEvaluator(series_eval_repo)
        confusion_mat = evaluator.draw_confusion_matrix(predictor.all_pred, predictor.all_gt, predictor.tag_dict)
        evaluator.draw_curve_pre_cls(predictor.all_pred.flatten(), predictor.all_gt.flatten(),
                                     f"{evaluator.SeriesName} Overall")
    return confusion_mat


if __name__ == "__main__":
    # eval
    root = r'...\FullEpisodeTriplets'
    labels_root = os.path.join(root, 'Evaluation')
    output_root = os.path.join(root, 'Output_Iteration2.3')

    series = os.listdir(labels_root)
    for ser in series:
        ser_path = os.path.join(labels_root, ser)
        output_ser = recreate_dir(output_root, ser)
        try:
            conf_mat = pred_classification_project(ser_to_project_id[ser], ser_to_iter[ser], ser_path, output_ser)
        except Exception as e:
            eprint('Failed analyzing series {} with exception:\n{}'.format(ser, e), e)
    print('Evaluation is done!!!')
