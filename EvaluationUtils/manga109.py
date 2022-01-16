from Animator.consolidation_api import BoundingBox, CharacterDetectionOutput, CharacterBoundingBox
from Animator.utils import eprint, to_json
import xml.etree.cElementTree as etree
from EvaluationUtils.vision_metrics import CVMetrics
import os
from shutil import copyfile, rmtree
import shutil
import cv2
import numpy as np
import random
import traceback
import json


seed = 1234567
random.seed(seed)
np.random.seed(seed)
eps = np.finfo(float).resolution


class Body:
    def __init__(self, bid, x, y, w, h, char):
        self.BoundingBox = BoundingBox(x, y, w, h)
        self.Id = bid
        self.Character = char


class Page:
    def __init__(self, page_dict):
        index = page_dict['index']
        width = page_dict['width']
        height = page_dict['height']
        self.Index = index
        self.Width = width
        self.Height = height
        self.Bodies = dict()

    def add_body(self, body_dict):
        if body_dict['id'] in self.Bodies:
            return

        xmin = int(body_dict['xmin'])
        xmax = int(body_dict['xmax'])
        ymin = int(body_dict['ymin'])
        ymax = int(body_dict['ymax'])

        body = Body(body_dict['id'], xmin, ymin, xmax-xmin, ymax-ymin, body_dict['character'])
        self.Bodies[body_dict['id']] = body
        return


class Character:
    def __init__(self, c_dict):
        self.Id = c_dict['id']
        self.Name = c_dict['name']


class Book:
    def __init__(self, xml_path):
        self.XmlPath = xml_path
        self.Root = self.parse_xml()
        title = self.Root.attrib['title']
        self.Title = title
        self.Characters = dict()
        self.Pages = dict()

        for child in self.Root:
            if child.tag == 'characters':
                for char_dict in child.iter():
                    if len({'id', 'name'} & set(char_dict.keys())) < 2:
                        continue
                    char = Character(char_dict.attrib)
                    self.Characters[char.Id] = char

            if child.tag == 'pages':
                page = None
                for page_dict in child.iter():
                    if page_dict.tag == 'page' and len({'index', 'width', 'height'} & set(page_dict.keys())) == 3:
                        page = Page(page_dict.attrib)
                        self.Pages[page.Index] = page
                        for label in page_dict.iter():
                            if label.tag == 'body' \
                                    and len({'id', 'xmin', 'ymin', 'xmax', 'ymax', 'character'} & set(label.keys()))==6\
                                    and page:
                                page.add_body(label.attrib)

    def parse_xml(self):
        xml_parser = etree.XMLParser(encoding="utf-8")
        target_tree = etree.parse(self.XmlPath, parser=xml_parser)
        xml_root = target_tree.getroot()
        return xml_root

    def add_character(self, cid, char):
        if cid in self.Characters:
            return
        self.Characters[cid] = char
        return

    def to_dict(self):
        final_dict = dict()
        for _, page in self.Pages.items():
            page_index = int(page.Index)
            final_dict[page_index] = dict()
            for _, body in page.Bodies.items():
                final_dict[page_index][body.Id] = body.BoundingBox
        return final_dict


def parse_manga(detections_file_path, labels_xml_path, min_confidence):
    comic_book_detections = CharacterDetectionOutput.read_from_json(detections_file_path)
    comic_book_gt = Book(labels_xml_path)
    detections = dict()
    for bbox in comic_book_detections.CharacterBoundingBoxes:
        if bbox.Confidence < min_confidence:
            continue
        page_index = int(bbox.KeyframeThumbnailId)
        if page_index in detections:
            detections[page_index][bbox.Id] = bbox.Rect
        else:
            detections[page_index] = {bbox.Id: bbox.Rect}

    ground_truth = comic_book_gt.to_dict()
    return ground_truth, detections


def evaluate_single_manga(detections_file_path, labels_xml_path, min_iou, min_conf):
    ground_truth, detections = parse_manga(detections_file_path, labels_xml_path, min_conf)
    precision, recall = CVMetrics.precision_recall_at_iou(ground_truth, detections, min_iou)
    return precision, recall


def evaluate_manga_109(min_iou, min_confidence):
    manga_repo = r'\..\Manga109\Manga109_released_2020_09_26'
    books_repo = os.path.join(manga_repo, 'images')
    labels_repo = os.path.join(manga_repo, 'annotations')
    comic_books_names = os.listdir(books_repo)
    precisions = []
    recalls = []
    running_count = 0
    for comic_book in comic_books_names:
        if running_count >= 11:
            break
        running_count += 1
        detections_json_path = os.path.join(books_repo, comic_book, 'Detections', 'animationdetectionoutput.json')
        gt_xml_path = os.path.join(labels_repo, f'{comic_book}.xml')
        book_precision, book_recall = evaluate_single_manga(detections_json_path, gt_xml_path, min_iou, min_confidence)
        precisions.append(book_precision)
        recalls.append(book_recall)
    return np.mean(precisions), np.mean(recalls)


def evaluate_detection_quality():
    ious_explored = [0.3]
    confidences_explored = [0.3, 0.35, 0.4, 0.45]
    results = dict()
    for iou_exp in ious_explored:
        for confidence_exp in confidences_explored:
            p, r = evaluate_manga_109(min_iou=iou_exp, min_confidence=confidence_exp)
            key = f'IoU={iou_exp}, DetConf={confidence_exp}'
            val = dict(Precision=p, Recall=r)
            results[key] = val
            print(f'Input: {key}; Output: {val}')
    return results


def populate_detections_with_gt_single_book(detections_input_json_path, detections_output_json_path,
                                            output_false_negatives_json_path, gt_xml_path, min_iou, min_confidence,
                                            begin_end_frame):
    # parse detections and labels
    comic_book_detections = CharacterDetectionOutput.read_from_json(detections_input_json_path)
    comic_book_gt = Book(gt_xml_path)
    gt_unassigned_labels = {p.Index: p.Bodies for _, p in comic_book_gt.Pages.items()}

    # join significant detections with its label
    for char_bbox in comic_book_detections.CharacterBoundingBoxes:
        page_index = int(char_bbox.KeyframeThumbnailId)

        # ignore the sides as they are not labeled
        if page_index < begin_end_frame or len(comic_book_gt.Pages) - page_index < begin_end_frame:
            gt_unassigned_labels[char_bbox.KeyframeThumbnailId] = dict()
            char_bbox = None
            continue

        # ignore low confidence detections
        if char_bbox.Confidence < min_confidence:
            char_bbox = None
            continue

        # populate the labels in the matched detection
        gt_bodies_in_page = [body for i, body in gt_unassigned_labels[str(page_index)].items()]
        gt_bboxes_in_page = [body.BoundingBox for body in gt_bodies_in_page]

        # check intersection gt vs detections
        if len(gt_bboxes_in_page) == 0:
            continue
        matching_label, ious = CVMetrics.matching_bbox_sets(gt_bboxes_in_page, [char_bbox.Rect], min_iou)
        matched_indices = list(matching_label)
        # case of no match
        if len(matching_label) == 0:
            continue
        # case of a bad match
        matched_index = matched_indices[0]
        matched_body = gt_bodies_in_page[matched_index]
        matched_iou = ious[matched_index][0]
        if matched_iou <= eps:
            continue

        # remove the gt instance from the unassigned list
        gt_unassigned_labels[str(page_index)].pop(matched_body.Id)

        # populate fields
        char_bbox.Character = matched_body.Character
        char_bbox.X_tsv = matched_body.BoundingBox.X
        char_bbox.Y_tsv = matched_body.BoundingBox.Y
        char_bbox.Width_tsv = matched_body.BoundingBox.Width
        char_bbox.Height_tsv = matched_body.BoundingBox.Height
        char_bbox.IoU = matched_iou
        char_bbox.IsLabeled = True

    # remove empty bboxes and empty unassigned gts
    comic_book_detections.CharacterBoundingBoxes = [det for det in comic_book_detections.CharacterBoundingBoxes if det]
    gt_unassigned_labels = {unassigned_gt_page_index: unassigned_gt_bodies
                            for unassigned_gt_page_index, unassigned_gt_bodies in gt_unassigned_labels.items()
                            if len(unassigned_gt_bodies) > 0}

    # serialize
    if os.path.isfile(detections_output_json_path):
        os.remove(detections_output_json_path)
    comic_book_detections.save_as_json(detections_output_json_path)

    # serialize the false positives
    if os.path.isfile(output_false_negatives_json_path):
        os.remove(output_false_negatives_json_path)
    to_json(gt_unassigned_labels, output_false_negatives_json_path)
    return


def populate_detections_with_gt_full_dataset(manga_repo, min_iou, min_confidence, begin_end_frame_position):
    books_repo = os.path.join(manga_repo, 'images')
    labels_repo = os.path.join(manga_repo, 'annotations')
    output_repo_for_clustering = os.path.join(manga_repo, 'clustering_input_files')
    comic_books_names = os.listdir(books_repo)
    running_count = 0
    for comic_book in comic_books_names:
        if running_count >= 11:
            break
        running_count += 1
        detections_json_path = os.path.join(books_repo, comic_book, 'Detections', 'animationdetectionoutput.json')
        gt_xml_path = os.path.join(labels_repo, f'{comic_book}.xml')
        output_labeled_detections_json_path = os.path.join(output_repo_for_clustering, f'{comic_book}.json')
        output_false_negatives_json_path = os.path.join(output_repo_for_clustering, f'{comic_book}_fn.json')
        try:
            populate_detections_with_gt_single_book(detections_json_path, output_labeled_detections_json_path,
            output_false_negatives_json_path, gt_xml_path, min_iou, min_confidence, begin_end_frame_position)
        except Exception as e:
            traceback.print_exc()
            eprint(' with exception: \'{}\'' % e)
            raise e


def draw_single_bbox(img, x, y, w, h, color, line_width):
    point1 = (x, y)
    point2 = (x + w, y + h)
    cv2.rectangle(img, point1, point2, color, line_width)
    return


def draw_frame_with_bboxes_and_labels(input_frame_path, frame_bboxes, frame_false_positives, frame_visualization_path):
    # read image and draw bboxes by NMS
    line_width = 2
    detection_color = (0, 0, 255)
    label_color = (0, 255, 0)
    fn_color = (255, 0, 0)
    img = cv2.imread(input_frame_path, cv2.IMREAD_COLOR)
    for detection in frame_bboxes:
        # draw the detection bbox visualization
        draw_single_bbox(img, detection.Rect.X, detection.Rect.Y, detection.Rect.Width, detection.Rect.Height, detection_color, line_width)
        # draw the gt if exist...
        if detection.IsLabeled:
            draw_single_bbox(img, detection.X_tsv, detection.Y_tsv, detection.Width_tsv, detection.Height_tsv, label_color, line_width)

    # draw fns if exist
    if frame_false_positives:
        for _, missed_gt_box in frame_false_positives.items():
            fn_bb = missed_gt_box['boundingBox']
            draw_single_bbox(img, fn_bb['x'], fn_bb['y'], fn_bb['width'], fn_bb['height'], fn_color, line_width)
    cv2.imwrite(frame_visualization_path, img)
    return


def sample_frames_and_draw_bboxes_single_book(original_frames_repo, detections_json_path, output_visualization_repo,
                                              false_negatives_json_path, min_iou, min_confidence,
                                              begin_end_frame_position):
    num_samples_per_book = 20

    # parse detections with labels
    comic_book_detections = CharacterDetectionOutput.read_from_json(detections_json_path)

    # parse false negatives
    with open(false_negatives_json_path, "r") as text_file:
        fn_gt = json.load(text_file)

    # create a repo
    if os.path.isdir(output_visualization_repo):
        shutil.rmtree(output_visualization_repo)
    os.mkdir(output_visualization_repo)

    # sample frames
    frame_indices = {f.KeyframeThumbnailId for f in comic_book_detections.CharacterBoundingBoxes if f}
    max_index = np.max([int(f) for f in frame_indices])
    valid_frames = [f for f in frame_indices if begin_end_frame_position < int(f) < max_index-begin_end_frame_position]
    sampled_frames = set(np.random.choice(valid_frames, num_samples_per_book, replace=False))

    # enumerate for book keeping
    frame_pool = dict()
    for bbox in comic_book_detections.CharacterBoundingBoxes:
        # filter those that weren't sampled
        if bbox.KeyframeThumbnailId not in sampled_frames:
            continue

        # book keeping of sampled frames
        if bbox.KeyframeThumbnailId in frame_pool:
            frame_pool[bbox.KeyframeThumbnailId].append(bbox)
        else:
            frame_pool[bbox.KeyframeThumbnailId] = [bbox]

    # enumerate and draw
    for frame_name_no_extention, frame_bboxes in frame_pool.items():
        frame_file_name = f'{frame_name_no_extention}.jpg'
        input_frame_path = os.path.join(original_frames_repo, frame_file_name)
        frame_visualization_path = os.path.join(output_visualization_repo, frame_file_name)
        frame_fns = fn_gt.get(str(int(frame_name_no_extention)), None)
        draw_frame_with_bboxes_and_labels(input_frame_path, frame_bboxes, frame_fns, frame_visualization_path)

    return


def sample_frames_and_draw_bboxes_full_dataset(manga_repo, min_iou, min_confidence, begin_end_frame_position):
    books_repo = os.path.join(manga_repo, 'images')
    dataset_visualization_repo = os.path.join(manga_repo, 'visualization')
    if not os.path.isdir(dataset_visualization_repo):
        os.mkdir(dataset_visualization_repo)
    labeled_detections = os.path.join(manga_repo, 'clustering_input_files')
    comic_books_names = os.listdir(books_repo)

    running_count = 0
    for comic_book in comic_books_names:
        if running_count >= 11:
            break
        running_count += 1
        detections_json_path = os.path.join(labeled_detections, f'{comic_book}.json')
        fn_json_path = os.path.join(labeled_detections, f'{comic_book}_fn.json')
        original_frames_repo = os.path.join(books_repo, comic_book)
        book_visualization_repo = os.path.join(dataset_visualization_repo, comic_book)
        try:
            sample_frames_and_draw_bboxes_single_book(original_frames_repo, detections_json_path,
                                                      book_visualization_repo, fn_json_path, min_iou, min_confidence,
                                                      begin_end_frame_position)
        except Exception as e:
            traceback.print_exc()
            eprint(' with exception: \'{}\'' % e)
            stop = 1
            raise e


def run_grouper_multiple():
    root_dir = r'\..\Manga109\Manga109_released_2020_09_26\images'
    test_books_names = ['AisazuNihaIrarenai',
                        'AkkeraKanjinchou',
                        'Akuhamu',
                        'AosugiruHaru',
                        'AppareKappore',
                        'Arisa',
                        'ARMS',
                        'BakuretsuKungFuGirl',
                        'Belmondo',
                        'BEMADER_P',
                        'BokuHaSitatakaKun']
    for book_name in test_books_names:
        book_detections_path = os.path.join(root_dir, book_name, 'Detections', 'animationdetectionoutput.json')


if __name__ == '__main__':
    minimum_intersection_over_union = 0.5
    min_conf = 0.3
    begin_end_frame = 5
    m_repo = r'\..\Manga109\Manga109_released_2020_09_26'
    # sample_frames_and_draw_bboxes_full_dataset(m_repo, minimum_intersection_over_union, min_conf, begin_end_frame)
    # populate_detections_with_gt_full_dataset(m_repo, minimum_intersection_over_union, min_conf, begin_end_frame)
    stop = 1
    jsons_repo = r'\..\Manga109\Manga109_released_2020_09_26\clustering_input_files'
    working_folder = r'\..\Manga109\Manga109_released_2020_09_26\Eval'
    files = [f for f in os.listdir(jsons_repo) if not f.endswith('_fn.json')]
    for json_file in files:
        repo_name = json_file.replace('.json', '')
        current_json_repo_path = os.path.join(working_folder, repo_name)
        detections_path = os.path.join(current_json_repo_path, 'animationdetectionoutput.json')
        source_file_path = os.path.join(jsons_repo, json_file)
        if os.path.isdir(current_json_repo_path):
            rmtree(current_json_repo_path)
        os.mkdir(current_json_repo_path)
        copyfile(source_file_path, detections_path)
