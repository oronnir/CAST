from bisect import bisect_right
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import wandb
from torch.optim import lr_scheduler

from Teacher.modeller.meters import AverageMeter
from Teacher.torcher.sigmoid_cross_entropy_loss_with_balancing import SigmoidCrossEntropyLossWithBalancing


def get_criterion(loss_type=False, multi_label_negative_sample_weights_file=None, cross_entropy_weights=None,
                  margin=None):
    """
    gets the loss criterion for training a classifier/featurizer
    :param loss_type: either 'multi-label', 'cross-entropy', or 'triplets'
    :param multi_label_negative_sample_weights_file:
    :param cross_entropy_weights:
    :return: the criterion layer
    :param margin: (optional) the triplets loss margin
    """
    criterion = None
    if loss_type is 'multi-label':
        if not multi_label_negative_sample_weights_file:
            print("Use BCEWithLogitsLoss")
            criterion = nn.BCEWithLogitsLoss().cuda()
        else:
            print("Use SigmoidCrossEntropyLossWithBalancing")
            with open(multi_label_negative_sample_weights_file, "r") as f:
                weights = [float(line) for line in f]
                criterion = SigmoidCrossEntropyLossWithBalancing(np.array(weights)).cuda()
    elif loss_type is 'cross-entropy':
        print("Use CrossEntropyLoss")
        if cross_entropy_weights:
            cross_entropy_weights = torch.tensor(cross_entropy_weights)
        criterion = nn.CrossEntropyLoss(weight=cross_entropy_weights).cuda()
    elif loss_type is 'triplets':
        print("Use TripletsLoss")
        criterion = nn.TripletMarginLoss(margin=margin, p=2).cuda()
    return criterion


def get_init_lr(args):
    if args.start_epoch == 0:
        return args.lr
    if args.lr_policy.lower() == 'step':
        lr = args.lr * args.gamma ** (args.start_epoch // args.step_size)
    elif args.lr_policy.lower() == 'multistep':
        milestones = [int(m) for m in args.milestones.split(',')]
        lr = args.lr * args.gamma ** bisect_right(milestones, args.start_epoch)
    elif args.lr_policy.lower() == 'exponential':
        lr = args.lr * args.gamma ** args.start_epoch
    elif args.lr_policy.lower() == 'plateau':
        assert args.start_epoch == 0, 'cannot resume training for plateau'
        lr = args.lr
    else:
        raise ValueError('Unknown lr policy: {}'.format(args.lr_policy))
    return lr


def set_default_hyper_parameter(args):
    args.lr = 0.1
    args.momentum = 0.9
    args.weight_decay = 1e-3
    args.lr_policy = 'STEP'
    args.step_size = 30
    args.gamma = 0.1
    args.optimizer = 'SGD'


def get_optimizer(model, args):
    # use default parameter for reproducible network
    if not args.force:
        print('Use default hyper parameter')
        set_default_hyper_parameter(args)

    init_lr = get_init_lr(args)
    print('initial learning rate: %f' % init_lr)

    if args.finetune:
        group_pretrained = []
        group_new = []
        for name, param in model.named_parameters():
            if 'fc' in name:
                group_new.append(param)
            else:
                group_pretrained.append(param)
        assert len(list(model.parameters())) == len(group_pretrained) + len(group_new)
        groups = [dict(params=group_pretrained, lr=args.lr*0.01, initial_lr=init_lr*0.01),
                  dict(params=group_new,  lr=args.lr, initial_lr=init_lr)]
    else:
        if args.start_epoch > 0:
            groups = [dict(params=list(model.parameters()), initial_lr=init_lr)]
        else:
            groups = model.parameters()

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(groups, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'Unknown Optimizer: {args.optimizer}')

    return optimizer


def get_scheduler(optimizer, args):
    if args.lr_policy.lower() == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma,
                                        last_epoch=args.start_epoch - 1)
    elif args.lr_policy.lower() == 'multistep':
        milestones = [int(m) for m in args.milestones.split(',')]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma,
                                             last_epoch=args.start_epoch - 1)
    elif args.lr_policy.lower() == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma,
                                               last_epoch=args.start_epoch - 1)
    elif args.lr_policy.lower() == 'plateau':
        assert args.start_epoch == 0, 'cannot resume training for plateau'
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=True)
    elif args.lr_policy.lower() == 'constant':
        scheduler = None
    else:
        raise ValueError('Unknown lr policy: {}'.format(args.lr_policy))

    return scheduler


def train(args, train_loader, featurizer_model, criterion, optimizer, epoch, logger, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    running_loss = []

    end = time.time()
    tic = time.time()
    for i, (anc, pos, neg) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # adding the triplets
        anchor_img = anc.to(device)
        positive_img = pos.to(device)
        negative_img = neg.to(device)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        anchor_out = featurizer_model(anchor_img)
        positive_out = featurizer_model(positive_img)
        negative_out = featurizer_model(negative_img)

        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), anc.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        running_loss.append(loss.cpu().detach().numpy())

        if i % args.print_freq == 0:
            speed = args.print_freq * args.batch_size / float(args.world_size) / (time.time() - tic)
            info_str = 'Epoch: [{0}][{1}/{2}]\t' \
                       'Speed: {speed:.2f} samples/sec\t' \
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, len(train_loader), speed=speed, batch_time=batch_time,
                        data_time=data_time, loss=losses)
            logger.info(info_str)
            tic = time.time()

            # 4. Log metrics to visualize performance
            wandb.log({"Triplets_Loss": loss})
    return running_loss


def get_labels_hist(imdb, normalize=False):
    labels_hist = []
    for i, label in enumerate(imdb.iter_cmap()):
        keys = list(imdb.iter_label(label))
        labels_hist.append(len(keys))
    if normalize:
        total_labels = sum(labels_hist)
        for i in range(len(labels_hist)):
            labels_hist[i] /= float(total_labels)

    return labels_hist


def get_balance_weights(imdb):
    labels_hist = get_labels_hist(imdb, normalize=True)
    balance_weights = []
    temperature = 1.0 / len(labels_hist)
    for i, count in enumerate(labels_hist):
        balance_weights.append(0 if count == 0 else max(temperature, np.exp(-count ** 2 / (2 * temperature ** 2))))
    balance_weights[-2] = 0.99
    sum_all = sum(balance_weights)
    for i, count in enumerate(labels_hist):
        balance_weights[i] /= sum_all
    return balance_weights
