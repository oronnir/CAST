import argparse
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import wandb

from Teacher.modeller import dist_utils as dist_utils
from Teacher.modeller.logger import Logger
from Teacher.torcher.classifier_dataloaders import triplet_data_loader
from Teacher.torcher.classifier_train_utils import get_criterion, get_optimizer, get_scheduler, train
from Teacher.torcher.custom_resnet import ResNetWithFeatures
from Teacher.torcher.weights_init import dense_layers_init

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def save_checkpoint(state, prefix, epoch, output_dir, is_best=False):
    filename = os.path.join(output_dir, '%s-%04d.pth.tar' % (prefix, epoch))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(output_dir, 'model_best.pth.tar'))


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # necessary inputs
    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('--labelmap', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=model_names + ['False'],
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('--gpu-id', default='0', type=str, help='The GPU id (default: "0")')
    parser.add_argument('--session-id', type=str, help='The session id')
    parser.add_argument('--episode', type=str, help='The episode name')
    parser.add_argument('--margin', default='0', type=float, help='The triplets loss margin (default: "0")')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # has default hyper parameter for ResNet
    parser.add_argument('--epochs', default=161, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--optimizer', default='SGD', type=str,
                        help='The torch.optim optimizer ("SGD", "AdamW")')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--is_multi_label', default=False, action='store_true')
    parser.add_argument('--ccs_loss_param', default=0.0, type=float)
    # distributed training
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', help='local_rank', required=False)
    parser.add_argument('--dist_url', default="tcp://127.0.0.1:2345",
                        help='dist_url')
    parser.add_argument('--distributed', default=False, action='store_true',
                        help='specify if you want to use distributed training (default=False)',
                        required=False)
    # need setup output dir
    parser.add_argument('--output-dir', default='./outputs/resnet18', type=str,
                        help='path to save checkpoint and log (default: ./outputs/resnet18)')
    parser.add_argument('--prefix', default=None, type=str,
                        help='model prefix (default: same with model names)')
    # Optimization setting
    parser.add_argument('--balance', action='store_true',
                        help='balance cross entropy weights')
    parser.add_argument('--upsample_factor', default=1, type=int, metavar='N',
                        help='upsample dataset by factor (default: 1 ),it is useful for small dataset')
    parser.add_argument('--lr-policy', default='STEP', type=str,
                        help='learning rate decay policy: STEP, MULTISTEP, EXPONENTIAL, PLATEAU, CONSTANT '
                             '(default: STEP)')
    parser.add_argument('--step-size', default=30, type=int,
                        help='step size for STEP decay policy (default: 30)')
    parser.add_argument('--milestones', default='30,60,90', type=str,
                        help='milestones for MULTISTEP decay policy (default: 30,60,90)')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='gamma for learning rate decay (default: 0.1)')
    parser.add_argument('--neg', dest='neg_weight_file', default=None,
                        help='weights of negative samples used in multi-label training. If specified, balanced loss'
                             ' will be used, otherwise, BCELoss will be used.')
    # force using customized hyper parameter
    parser.add_argument('-f', '--force', dest='force', action='store_true',
                        help='force using customized hyper parameter')
    parser.add_argument('--finetune', dest='finetune', action='store_true',
                        help='finetune last layer by using 0.1x lr for previous layers')
    # display
    parser.add_argument('--snapshot_freq', default=20, type=int,
                        help='snapshot frequency in epochs (default: 20)')
    return parser


def create_pretrained(args, logger, num_classes):
    if args.pretrained:
        logger.info("=> using pre-trained model '{}'".format(args.arch))
        if args.arch.startswith('resnet'):
            from torchvision.models.resnet import model_urls
        elif args.arch.startswith('alexnet'):
            from torchvision.models.alexnet import model_urls
        elif args.arch.startswith('vgg'):
            from torchvision.models.vgg import model_urls
        model_urls[args.arch] = model_urls[args.arch].replace('https://', 'http://')
        model = models.__dict__[args.arch](pretrained=True)
        if args.arch.startswith('alexnet'):
            classifier = list(model.classifier.children())
            model.classifier = nn.Sequential(*classifier[:-1])
            model.classifier.add_module(
                '6', nn.Linear(classifier[-1].in_features, num_classes))
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            dense_layers_init(model)

    else:
        logger.info("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=num_classes)

    if args.ccs_loss_param > 0:
        model = ResNetWithFeatures(model)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = dist_utils.DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    return model


def fine_tune_triplets_session(raw_args=None, model=None):
    args = get_parser().parse_args(raw_args)
    seed = 1234567
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device(f'cuda:{args.gpu_id}')
    cuda_report = torch.cuda.memory_summary(device=None, abbreviated=False)
    print(cuda_report)

    if device.type == "cuda":
        device_brand = torch.cuda.get_device_name()
        print(f'Running with device: "{device_brand}" on id: "{args.gpu_id}"')

    if args.distributed:
        args.local_rank = dist_utils.env_rank()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method=args.dist_url, rank=dist_utils.env_rank(),
                                world_size=dist_utils.env_world_size())
        assert (dist_utils.env_world_size() == dist.get_world_size())  # check if there are

    logger = Logger(args.output_dir, args.prefix)
    logger.info('distributed? {}'.format(args.distributed))

    if args.local_rank == 0:
        logger.info('called with arguments: {}'.format(args))

    # Data loading code
    data_loader = triplet_data_loader(args.data, batch_size=args.batch_size, num_workers=args.workers,
                                      distributed=args.distributed)

    criterion = get_criterion('triplets', margin=.5)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    cudnn.benchmark = True
    if args.distributed:
        dist_utils.sum_tensor(torch.tensor([1.0]).float().cuda())

    # w&b
    # login
    wandb.login(key='???')
    wandb_run = wandb.init(project='???', entity='???', name=args.session_id, reinit=True)

    try:
        # Save model inputs and hyperparameters
        config = wandb.config
        config.session_id = args.session_id
        config.optimizer = args.optimizer
        config.episode = args.episode
        config.learning_rate = args.lr
        config.momentum = args.momentum
        config.gamma = args.gamma
        config.start_epoch = args.start_epoch
        config.epochs = args.epochs
        config.batch_size = args.batch_size
        config.triplets_margin = args.margin
        config.weight_decay = args.weight_decay

        # Log gradients and model parameters
        wandb.watch(model)

        for epoch in range(args.start_epoch, args.epochs):
            epoch_tic = time.time()
            if args.distributed:
                data_loader.sampler.set_epoch(epoch)

            scheduler.step()
            losses = train(args, data_loader, model, criterion, optimizer, epoch, logger, device)
            print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, args.epochs, np.mean(losses)))

            if args.local_rank == 0 and epoch % args.snapshot_freq == 0 or epoch == args.epochs - 1:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'multi_label': args.is_multi_label,
                }, args.prefix, epoch + 1, args.output_dir)
                info_str = 'Epoch: [{0}]\t' \
                           'Time {time:.3f}\t'.format(epoch, time=time.time() - epoch_tic)
                logger.info(info_str)

    except Exception as e:
        print(f'Failed with exception: {e}', e)
        raise e
    finally:
        wandb_run.finish()
