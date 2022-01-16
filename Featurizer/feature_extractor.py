import os

import cv2
import numpy as np
import PIL.Image as Image
import six
import torch
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from torch import nn
from Featurizer.seresnext import SEResNext
from Teacher.modeller.triplet_tsv import TripletTsv


class ImageFeaturizer(object):
    """Class for extracting features
        Parameters:
        model_path: str, Path to customized model, default None
        model_name: str, Either 'seresnext', 'art_miner', or 'None', default None
        labelmap_trained: str, provide if .pth file is given as a model, default None
        arch: str, default resnet-18
    """

    def __init__(self, model_path=None, model_name=None, labelmap_trained=None, arch='resnet18', gpu_id=None,
                 mode='eval'):
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

        self.model_path = model_path
        self.model_name = model_name
        self.labelmap_trained = labelmap_trained
        self.arch = arch
        self.set_model()
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

    def set_model(self):
        """model setup, use single process for now"""
        if self.model_path:
            num_classes = None
            if self.labelmap_trained:
                cmap_trained = self.load_labelmap_list()
                num_classes = len(cmap_trained)
            if self.model_name == 'art_miner':
                num_classes = 1000
            self.model = self.load_custom_model(num_classes=num_classes)
        else:
            self.model = self.load_pretrained_model()

    def extract_features(self, path_images, exts=['jpg', 'png'], batch_size=32, num_workers=4, gpu_id=None):
        """User should call this method to extract features
            Parameters:
            path_images: str, path to folder containing images or path to single image
            exts: valid extensions for an image file, default ['png', 'jpg']
            batch_size', str, batch size, default 32
            num_workers', str, number of multiprocessing workers to load the data, default 4
        """

        sampled_batches = self.load_data(
            path_images=path_images,
            extensions=exts,
            transforms=self.get_transforms(),
            batch_size=batch_size,
            num_workers=num_workers)

        features_dict = self.feature_extraction(sampled_batches, gpu_id)
        image_ids = list(features_dict.keys())
        image_ids.sort()
        features = np.zeros((len(image_ids), len(features_dict[image_ids[0]])))

        for i in range(len(image_ids)):
            features[i] = features_dict[image_ids[i]]

        return features, image_ids

    @staticmethod
    def get_transforms():
        """Transformations to be applied on images"""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    @staticmethod
    def load_data(path_images, extensions, transforms, batch_size, num_workers):
        """
        Load the images, apply transformations and creates batches

        Parameters:
        :path_images: str, path to folder containing images or path to single image
        :extensions: valid extensions for an image file, default ['png', 'jpg']
        :batch_size', str, batch size, default 32
        :num_workers', str, number of multiprocessing workers to load the data, default 4
        """
        feature_dataset = FeatureExtractionDataset(
            path_images=path_images,
            exts=extensions,
            transform=transforms,
        )

        return DataLoader(
            feature_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)

    def feature_extraction(self, sampled_batches, gpu_id=None):
        """Extract features on the given set of batches"""
        feature_dict = {}
        cpu_device = torch.device("cpu")
        gpu_device_id = '' if gpu_id is None else f':{gpu_id}'
        gpu_device = torch.device(f"cuda" + gpu_device_id)
        for i, batch in enumerate(tqdm(sampled_batches)):
            images = batch['image']
            image_ids = batch['key']
            images = images.to(gpu_device)
            with torch.no_grad():
                if self.model_name == 'seresnext':
                    features = self.model.features(images)
                else:
                    features = self.model(images)
                features = features.squeeze()
                features = features.to(cpu_device).numpy()
            feature_dict.update(
                {img_id: fea for img_id, fea in zip(image_ids, features)}
            )
        print("Feature extraction done!")
        return feature_dict

    def load_pretrained_model(self):
        """this function loads  pre-trained model from PyTorch model zoo"""

        arch = self.arch
        print("=> using ImageNet pre-trained model '{}'".format(arch))
        model = models.__dict__[arch](pretrained=True)
        layers = list(model.children())[:-1]  # remove last FC layer
        model = torch.nn.Sequential(*layers)
        print(model)
        model = torch.nn.DataParallel(model).cuda()
        return model

    def load_custom_model(self, num_classes=None):
        """this function loads custom model given a path to model snapshot"""

        model_path = self.model_path
        model_name = self.model_name
        arch = self.arch
        checkpoint = torch.load(model_path)

        arch = checkpoint['arch'] if 'arch' in checkpoint else arch
        num_classes = checkpoint['num_classes'] if 'num_classes' in checkpoint else num_classes
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        if model_name is 'seresnext':
            model = SEResNext(num_classes=num_classes)
            model.load_feature_extractor_state(state_dict)
            model = model.cuda()
        else:
            model = models.__dict__[arch](num_classes=num_classes)
            model.load_state_dict(state_dict)

            layers = list(model.children())[:-1]  # remove last FC layer
            model = torch.nn.Sequential(*layers)
            model = torch.nn.DataParallel(model).cuda()

        if 'epoch' in checkpoint:
            print("=> loaded checkpoint '{}' (epoch {})".format(model_path, checkpoint['epoch']))
        else:
            print("=> loaded checkpoint '{}'".format(model_path))

        print(model)
        return model

    def load_labelmap_list(self):
        """Reads and return the labelmap file"""
        labelmap = []
        with open(self.labelmap_trained, encoding='utf-8') as fin:
            labelmap += [six.text_type(line.rstrip()) for line in fin]
        return labelmap


class FeatureExtractionDataset(Dataset):
    """Feature Extraction dataset."""

    def __init__(self, path_images, transform, exts):
        if os.path.isdir(path_images):
            self.filenames = [os.path.join(path_images, name) for name in os.listdir(path_images) if
                              name.split('.')[-1] in exts]
        elif os.path.isfile(path_images) and path_images.split('.')[-1] in exts:
            self.filenames = [path_images]
        else:
            raise ValueError('Invalid Path')

        self.filenames.sort()
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_name = self.filenames[idx]
        image = cv2.imread(image_name)
        image = image[:, :, (2, 1, 0)]  # BGR to RGB
        image = Image.fromarray(image, mode='RGB')  # save in PIL format
        if self.transform:
            image = self.transform(image)

        sample = {'key': image_name, 'image': image}

        return sample


class TripletsDataset(Dataset):
    """Feature Extraction based triplets loss dataset."""

    @staticmethod
    def pad(im):
        width = im.shape[0]
        height = im.shape[1]
        padding_func = nn.ConstantPad2d((256 - width, 256 - height), 0)
        return padding_func(torch.tensor(im))

    def __init__(self, tsv_path, transform=None):
        self.tsv_path = tsv_path
        self.row_order = ['anchor', 'positive', 'negative']
        self.transform = transform
        self._data = TripletTsv(tsv_path)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        if self.transform is not None:
            return [self.transform()(im) for im in self._data[index]]
        return self._data[index]
