from torchvision import transforms

MEANS = (104.0, 117.0, 123.0)
TRAIN_CANVAS_SIZE = (416, 416)
TEST_CANVAS_SIZE = (416, 416)
MAX_BOXES = 30
USE_DARKNET_LIB = True
NO_FLIP = 0
RANDOM_FLIP = 0.5
SCALE_RANGE = (0.25, 2)


__all__ = ['ClassifierTrainAugmentation']


class ClassifierTrainAugmentation(object):

    def __init__(self):
        self._set_augmentation_params()
    
    def __call__(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=self.scale),
            transforms.RandomAffine(degrees=self.rotation_upto_deg),
            transforms.ColorJitter(brightness=self.exposure, saturation=self.saturation, hue=self.hue),
            transforms.RandomHorizontalFlip(self.flip),
            transforms.ToTensor(),
            normalize,
        ])

    def _set_augmentation_params(self):
        self.hue = 0
        self.saturation = 1
        self.exposure = 1
        self.rotation_upto_deg = 45
        self.scale = (0.75, 1.5)
        self.flip = RANDOM_FLIP
