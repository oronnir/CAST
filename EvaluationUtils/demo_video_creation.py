from CustomVision.predict_ic import pred_classification_project
from Animator.bbox_grouper_api import CharacterDetectionOutput
import os
import shutil
import time
from Animator.utils import eprint
import traceback
import cv2
import pandas as pd


DETECTION_TH = 0.5


class NonMaxSuppressedFrame:
    def __init__(self, frame_id, exception_labels=None):
        self.Id = frame_id
        self.Exceptions = set(exception_labels) if exception_labels else set()
        self.SuppressedDetections = dict()
        self.NonSuppressedDetections = []

    def add(self, detection_candidate, class_name):
        if class_name in self.Exceptions:
            self.NonSuppressedDetections.append(detection_candidate)
            return

        if class_name in self.SuppressedDetections:
            # swap if the candidate is more confident
            if detection_candidate.Confidence > self.SuppressedDetections[class_name].Confidence:
                self.SuppressedDetections[class_name] = detection_candidate
        else:
            self.SuppressedDetections[class_name] = detection_candidate
        return

    def get_nms_detections(self):
        all_nms_detections_in_frame = self.NonSuppressedDetections[:]
        all_nms_detections_in_frame += self.SuppressedDetections.values()
        return all_nms_detections_in_frame


def non_max_suppression(detects, classes, exceptions=None):
    # map frames to bboxes
    frame_to_boxes = dict()
    for det in detects.CharacterBoundingBoxes:
        # skip low confidence detections
        if det.Confidence < DETECTION_TH or det.ThumbnailId not in classes:
            continue

        if det.KeyframeThumbnailId in frame_to_boxes:
            frame_to_boxes[det.KeyframeThumbnailId].add(det, classes[det.ThumbnailId])
        else:
            frame_to_boxes[det.KeyframeThumbnailId] = NonMaxSuppressedFrame(det.KeyframeThumbnailId, exceptions)
            frame_to_boxes[det.KeyframeThumbnailId].add(det, classes[det.ThumbnailId])

    return frame_to_boxes


if __name__ == '__main__':
    # keys
    ser_path = r"\..\Floogals\HighResFrames\Detections\animationdetectionoriginalimages"
    output_ser = r"\..\Floogals\HighResFrames\Detections\classificationOutput"
    high_conf_detections = r'\..\Floogals\HighResFrames\Detections\Unknown'
    frames_dir = r'\..\Floogals\HighResFrames'
    bboxed_frames_dir = r'\..\Floogals\NmsBboxedHighResFrames'

    project_id_arg = '???'
    iteration = 'Iteration1'
    ser = 'Floogals'

    # parse detections json
    detection_json_path = r"\..\Floogals\HighResFrames\Detections\animationdetectionoutput.json"
    detections = CharacterDetectionOutput.read_from_json(detection_json_path)

    if os.path.isdir(output_ser):
        shutil.rmtree(output_ser)
        time.sleep(2)
    os.mkdir(output_ser)
    if os.path.isdir(high_conf_detections):
        shutil.rmtree(high_conf_detections)
        time.sleep(2)
    os.mkdir(high_conf_detections)

    # enumerate bboxes, filter low confidence detections and classify
    for detection in detections.CharacterBoundingBoxes:
        if detection.Confidence < DETECTION_TH:
            print('Skipping low conf: {:.3f} on thumbnail id: {}'.format(detection.Confidence, detection.ThumbnailId))
            continue

        file_name = '{}.jpg'.format(detection.ThumbnailId)
        source = os.path.join(ser_path, file_name)
        target = os.path.join(high_conf_detections, file_name)
        shutil.copyfile(source, target)

    try:
        conf_mat = pred_classification_project(project_id_arg, iteration, high_conf_detections, output_ser, False)
    except Exception as e:
        eprint('Failed analyzing series {} with exception:\n{}'.format(ser, e))
        traceback.print_exc()

    # draw bboxes and predictions
    character_to_color = dict(Blue=(255, 0, 0), Flo=(0, 255, 255), Fleeker=(0, 0, 255), Boomer=(0, 255, 0),
                              Unknown=(255, 255, 255))
    line_width = 3

    # read classifications
    classifications = pd.read_csv(os.path.join(output_ser, 'Evaluation_classificationOutput.tsv'), sep='\t')
    thumbnail_to_classification = dict()
    for t in classifications.iterrows():
        row = t[1]
        key = row['Id'].split('\\')[-1].split('.')[0]
        thumbnail_to_classification[key] = row['Prediction']

    frame_to_consolidated_bboxes = non_max_suppression(detections, thumbnail_to_classification, exceptions=['Unknown'])

    # manage a directory with the drawn bboxes on the frames + labels
    if os.path.isdir(bboxed_frames_dir):
        shutil.rmtree(bboxed_frames_dir)
        time.sleep(2)
    os.mkdir(bboxed_frames_dir)

    for frame_name in os.listdir(frames_dir):
        frame_code = frame_name.split('_')[-1].split('.')[0]
        frame_path = os.path.join(frames_dir, frame_name)

        # skip directories
        if os.path.isdir(frame_path):
            continue

        # copy frame into the new dir if it simply has no detections
        target = os.path.join(bboxed_frames_dir, frame_name)
        if frame_code not in frame_to_consolidated_bboxes:
            shutil.copy(frame_path, target)
            continue

        # read image and draw bboxes by NMS
        img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        nms_detections = frame_to_consolidated_bboxes[frame_code].get_nms_detections()
        for detection in nms_detections:
            # skip low conf detections
            if detection.ThumbnailId not in thumbnail_to_classification:
                continue

            # set the bbox visualization
            character_name = thumbnail_to_classification[detection.ThumbnailId]
            character_color = character_to_color[character_name]
            point1 = (detection.Rect.X, detection.Rect.Y)
            point2 = (detection.Rect.X+detection.Rect.Width, detection.Rect.Y+detection.Rect.Height)
            cv2.rectangle(img, point1, point2, character_color, line_width)
            cv2.putText(img, character_name, (detection.Rect.X+2, detection.Rect.Y + detection.Rect.Height-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, character_color, 2, cv2.LINE_AA)
        cv2.imwrite(target, img)
