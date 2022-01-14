import argparse
import base64
import json
import os
import time
import uuid
from typing import List

import torch
from sklearn.metrics.pairwise import cosine_similarity

from Featurizer.feature_extractor import ImageFeaturizer
from Teacher.modeller.triplets_train_session import fine_tune_triplets_session
from Tracker.tracks import Triplet
from Animator.utils import eprint, create_dir_if_not_exist, flatten


class Featurizer:
    MODEL_REPO = r'/../data/featurizer_model/'

    def __init__(self, gpu_id, session_id, out_model_path, input_model_pth=None):
        model_pth = input_model_pth if input_model_pth else Featurizer.MODEL_REPO + 'epoch_40.pth'
        self.extractor = ImageFeaturizer(
            model_path=model_pth,
            model_name='seresnext',
            labelmap_trained=Featurizer.MODEL_REPO + 'labelmap3.txt',
            arch=False,
            gpu_id=gpu_id,
            mode='eval' if input_model_pth else 'train'
        )
        self.args = []
        self.session_id = session_id
        self.short_id = session_id[:8]
        self.finetuned_model_pth_path = out_model_path.replace('.pth', f'{self.short_id}.pth')
        self.finetuned_hyperparam_json_path = self.finetuned_model_pth_path.replace('.pth', '_hyperparams.json')

    @staticmethod
    def get_base_model(gpu_id, session_id, output_model_path):
        """returns a copy of the base model in train mode"""
        featurizer = Featurizer(gpu_id, session_id, output_model_path)
        return featurizer
    
    @staticmethod
    def get_finetuned_model(gpu_id, pth_path):
        """returns the model in pth in eval mode"""
        session_id = str(uuid.uuid4())
        featurizer = Featurizer(gpu_id, session_id, pth_path, pth_path)
        return featurizer

    def featurize(self, image_repo, num_workers=4, gpu_id=None):
        """runs the featurizer on the image_repo and returns features and image paths"""
        features, image_ids = self.extractor.extract_features(path_images=image_repo, num_workers=num_workers,
                                                              gpu_id=gpu_id)
        return features, image_ids

    @staticmethod
    def image_to_base64(im_path: str) -> str:
        with open(im_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return encoded_string.__repr__()

    @staticmethod
    def _triplets_to_tsv(image_repo: str, triplets: List[Triplet]) -> str:
        """
        take the sampled triplets and form a TSV file for the imdb
        :param image_repo: a repository with all cropped jpegs
        :param triplets: a list of all triplets
        :return: the path to the serialized TSV
        """
        tsv_path = os.path.join(image_repo, 'triplets.tsv')

        def thumbnail_to_path(thumbnail_id):
            im_name = f'{thumbnail_id}.jpg'
            return os.path.join(image_repo, im_name)

        tsv_writer = open(tsv_path, 'w')
        for triplet in triplets:
            anc_pos_neg = [thumbnail_to_path(triplet['anchor']),
                           thumbnail_to_path(triplet['positive']),
                           thumbnail_to_path(triplet['negative'])]
            for i, im_path in enumerate(anc_pos_neg):
                im_b64 = Featurizer.image_to_base64(im_path)
                tsv_writer.write(im_b64)
                if i < 2:
                    tsv_writer.write('\t')
            tsv_writer.write(os.linesep)
        tsv_writer.close()
        return tsv_path

    def serialize(self):
        """store the fine-tuned model with its args"""
        torch.save(self.extractor.model.state_dict(), self.finetuned_model_pth_path)
        print(f'The fine-tuned model was saved into {self.finetuned_model_pth_path}')

        with open(self.finetuned_hyperparam_json_path, 'w') as j_writer:
            json.dump(self.args, j_writer)
        print(f'Hyper-parameters json was saved into {self.finetuned_hyperparam_json_path}')

    def deserialize_args(self):
        """load only the args json and assume self is its corresponding model"""
        if not os.path.isfile(self.finetuned_hyperparam_json_path):
            exc_message = f'File not found: {self.finetuned_hyperparam_json_path}. The args json is missing...'
            raise Exception(exc_message)
        with open(self.finetuned_hyperparam_json_path, 'r') as j_reader:
            self.args = json.load(j_reader)
        print(f'Hyper-parameters json was loaded from {self.finetuned_hyperparam_json_path}')

    def fine_tune(self, image_repo: str, triplets: List[Triplet], config: dict):
        """
        copy the model, fine-tune, and return the model
        :param image_repo:
        :param triplets:
        :param config:
        :return:
        """
        # fine-tune
        try:
            start_time = time.time()
            data_tsv = Featurizer._triplets_to_tsv(image_repo, triplets)
            data_root, _ = os.path.split(image_repo)
            output_repo = create_dir_if_not_exist(data_root, 'model')
            args = ['--data', data_tsv, '--arch', str(self.extractor.arch), '--output-dir', output_repo, '--labelmap',
                    self.extractor.labelmap_trained, '--resume', self.extractor.model_path]
            session_configurations = flatten([[f'--{k.replace("_", "-")}', str(v)] for k, v in config.items()])
            args += list(session_configurations) + ['--force']
            self.args = args

            # execute
            fine_tune_triplets_session(args, self.extractor.model)
            print(f'the fine-tune took {time.time() - start_time:.2f} seconds')

            # serialize
            self.serialize()
        except Exception as e:
            eprint(f'fine-tune the featurizer has raised an exception {e}', e)
            return None
        return


def get_parser():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='extract features from imagenet pre-trained models')
    parser.add_argument('--path_images', required=True, help='path to folder containing images or path to single image')
    parser.add_argument('--exts', required=False, default=['jpg', 'jpeg', 'png'],
                        help='valid extensions for an image file')
    parser.add_argument('--batch_size', required=False, default=8, help='batch size')
    parser.add_argument('--num_workers', required=False, default=1,
                        help='number of multiprocessing workers to load the data')
    parser.add_argument('--shuffle', required=False, default=False, help='shuffle the data')

    parser.add_argument('--model_path', required=False, type=str, default=None,
                        help='Path to customized model')
    parser.add_argument('--model_name', default=None,
                        help="One of 'seresnext', 'art_miner', 'None'")
    parser.add_argument('--labelmap_trained', required=False, type=str, default=None,
                        help='provide if .pth file is given as a model')
    parser.add_argument('--arch', required=False, type=str, default='resnet18',
                        help='ImageNet pretrained model or Art-Miner uses resnet-18')
    parser.add_argument('--gpu_ids', required=False, type=str, default=None,
                        help='GPU IDs to be used')
    return parser


if __name__ == "__main__":
    repo = r'/???/data/featurizer_model/southpark_test_samples'
    feat = Featurizer('0', 'session?', 'output_model?')
    extracted_features, extracted_image_ids = feat.featurize(repo)
    cosines = cosine_similarity(extracted_features)
