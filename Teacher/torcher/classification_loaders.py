from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from Featurizer.feature_extractor import TripletsDataset
from Teacher.modeller.dist_utils import env_world_size, env_rank
from Teacher.torcher.augmentor import ClassifierTrainAugmentation


WRAPPING = True
USE_WEIGHTED_BALLANCE_SAMPLER = False
SEQUENTIAL_SAMPLER = False
RANDOM_SAMPLER = not SEQUENTIAL_SAMPLER
MIN_ITERS_PER_EPOCH = 10


def triplet_data_loader(datafile, batch_size=64, num_workers=2, distributed=True):
    """prepares data loader for training
    :param datafile: str, path to file with data
    :upsample_factor: int, number of times to sample the same region in epoch
    :param batch_size: int, batch size per GPU
    :param num_workers: int, number of workers per GPU
    :param distributed: bool, if distributed training is used
    :return: data loader
    """
    augmenter = ClassifierTrainAugmentation()
    augmented_dataset = TripletsDataset(tsv_path=datafile, transform=augmenter)
    sampler = DistributedSampler(augmented_dataset, num_replicas=env_world_size() if distributed else 1,
                                 rank=env_rank() if distributed else 0)

    return DataLoader(augmented_dataset, batch_size=batch_size,
                      sampler=sampler, num_workers=num_workers, pin_memory=True)
