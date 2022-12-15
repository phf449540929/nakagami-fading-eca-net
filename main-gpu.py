# -*- coding: UTF-8 -*-
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from torch.autograd import Variable
from sklearn import model_selection
import numpy
import datetime

from dataset_train import DatasetTrain
from pretreatment_feature_extraction import feature_extraction

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
'''
thread
default 16
'''
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int, nargs='+',
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--ksize', default=None, type=list,
                    help='Manually select the eca module kernel size')
parser.add_argument('--action', default='', type=str,
                    help='other information.')

best_prec1 = 0


def main():
    print(torch.cuda.is_available())
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)

    global args, best_prec1
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](k_size=args.ksize, pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.ksize == None:
            model = models.__dict__[args.arch]()
        else:
            model = models.__dict__[args.arch](k_size=args.ksize)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            # print(torch.cuda.is_available())
            '''
            gpu
            '''
            model = torch.nn.DataParallel(model).cuda()
            '''
            cpu
            '''
            # model = torch.nn.DataParallel(model).to('cpu')

    # print(model)

    # get the number of models parameters
    print('Number of models parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    root_dir = args.data[:-4]
    snr = root_dir.split('/')[-1]
    train_dir = root_dir + '-pre-train.csv'
    validate_dir = root_dir + '-pre-validate.csv'
    print(root_dir)
    # x, y = feature_extraction(root_dir)
    # x_train, x_valid, y_train, y_valid = model_selection.train_test_split(x, y)

    # traindir = os.path.join(args.data, 'train')
    # valdir = os.path.join(args.data, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    """
    dataset processing
    ---------------------------------------------------------------------------------------
    """
    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    #
    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    #     train_sampler = None

    dataset_train = DatasetTrain(train_dir)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    '''
    validate dataset
    '''
    dataset_validate = DatasetTrain(validate_dir)

    trans_list = dataset_validate.trans_list

    val_loader = torch.utils.data.DataLoader(
        dataset_validate, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)
    #
    if args.evaluate:
        m = time.time()
        _, _ = validate(val_loader, model, criterion, trans_list)
        n = time.time()
        print((n - m) / 3600)
        return

    directory = "runs/%s/" % (args.arch + '_' + args.action)
    if not os.path.exists(directory):
        os.makedirs(directory)

    """
    train
    ---------------------------------------------------------------------------------------
    """

    Loss_plot = {}
    train_prec1_plot = {}
    train_prec5_plot = {}
    val_prec1_plot = {}
    val_prec5_plot = {}

    acc_plot = {}
    acc_txt = "/acc-{}.txt".format(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S.%f'))

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        # train(train_loader, model, criterion, optimizer, epoch)
        loss_temp, train_prec1_temp, train_prec5_temp = train(train_loader, model, criterion, optimizer, epoch)
        Loss_plot[epoch] = loss_temp
        train_prec1_plot[epoch] = train_prec1_temp
        train_prec5_plot[epoch] = train_prec5_temp

        # evaluate on validation set
        # prec1 = validate(val_loader, model, criterion)
        prec1, prec5, acc_str = validate(val_loader, model, criterion, trans_list)
        val_prec1_plot[epoch] = prec1
        val_prec5_plot[epoch] = prec5

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        acc_plot[epoch] = acc_str

        # 将Loss,train_prec1,train_prec5,val_prec1,val_prec5用.txt的文件存起来
        data_save(directory + 'Loss_plot.txt', Loss_plot)
        data_save(directory + 'train_prec1.txt', train_prec1_plot)
        data_save(directory + 'train_prec5.txt', train_prec5_plot)
        data_save(directory + 'val_prec1.txt', val_prec1_plot)
        data_save(directory + 'val_prec5.txt', val_prec5_plot)

        fold = directory + snr
        if not os.path.exists(fold):
            os.makedirs(fold)

        data_save(fold + acc_txt, acc_plot)

        end_time = time.time()
        time_value = (end_time - start_time) / 3600
        print("-" * 80)
        print(time_value)
        print("-" * 80)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses_batch = {}
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        '''
        gpu
        '''
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        input = input.type(torch.cuda.FloatTensor)
        output = model(input)

        target = torch.tensor(target, dtype=torch.long)
        loss = criterion(output, target)

        '''
        cpu
        '''
        # output = model(input.to(torch.float32))
        # loss = criterion(output, torch.tensor(target).to(torch.long))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'

                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, trans_list):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()

        label = [0, 1, 2, 3, 4, 5]

        correct = list(0. for i in range(len(label)))
        total = list(0. for i in range(len(label)))

        for i, (input, target) in enumerate(val_loader):
            '''
            gpu
            '''
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            input = input.type(torch.cuda.FloatTensor)
            output = model(input)

            # print("input")
            # print(input)
            # print("output")
            # print(output)
            '''
            tensor([[ 0.2265,  0.2844,  0.2016,  ..., -0.0020,  0.0069, -0.0125],
                    [ 0.2265,  0.2844,  0.2016,  ..., -0.0020,  0.0069, -0.0125],
                    [ 0.2265,  0.2844,  0.2016,  ..., -0.0020,  0.0069, -0.0125],
                    ...,
                    [ 0.2265,  0.2844,  0.2016,  ..., -0.0020,  0.0069, -0.0125],
                    [ 0.2265,  0.2844,  0.2016,  ..., -0.0020,  0.0069, -0.0125],
                    [ 0.2265,  0.2844,  0.2016,  ..., -0.0020,  0.0069, -0.0125]],
                    device='cuda:0')
            '''


            target = torch.tensor(target, dtype=torch.long)
            loss = criterion(output, target)
            '''
            cpu
            '''
            # output = model(input.to(torch.float32))
            # loss = criterion(output, torch.tensor(target).to(torch.long))

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            prediction = torch.argmax(output, 1)
            # print("prediction:")
            # print(prediction)
            '''
            tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0')
            '''

            # print("target")
            # print(target)
            '''
            tensor([4, 2, 1, 4, 2, 4, 5, 4, 3, 2, 2, 0, 4, 0, 5, 0, 4, 4, 1, 4, 1, 1, 3, 4,
                    5, 4, 4, 3, 2, 3, 3, 2, 0, 5, 1, 4, 3, 4, 5, 1, 1, 2, 1, 1, 5, 4, 5, 0,
                    1, 3, 1, 5, 5, 5, 2, 2, 0, 3, 0, 2, 0, 4, 2, 2, 2, 5, 2, 5, 1, 4, 0, 5,
                    1, 5, 0, 5, 2, 0, 5, 2, 5, 1, 3, 3, 2, 4, 2, 1, 3, 4, 4, 0, 4, 0, 4, 2,
                    0, 0, 1, 5, 0, 2, 0, 0, 1, 5, 1, 5, 1, 3, 5, 3, 1, 3, 1, 2, 3, 2, 5, 3,
                    3, 5, 3, 5, 2, 2, 3, 5, 5, 1, 0, 3, 0, 4, 2, 4, 5, 2, 1, 0, 3, 4, 4, 1,
                    5, 5, 1, 4, 5, 3, 0, 4, 2, 4, 0, 0, 1, 0, 2, 2, 1, 2, 3, 1, 0, 3, 4, 2,
                    1, 4, 5, 4, 4, 2, 1, 2, 1, 2, 4, 0, 2, 5, 1, 0, 1, 4, 2, 1, 1, 3, 4, 5,
                    4, 0, 4, 5, 5, 4, 3, 4, 3, 5, 3, 4, 3, 3, 4, 1, 5, 2, 1, 3, 4, 1, 2, 2,
                    1, 0, 4, 5, 1, 0, 2, 5, 2, 5, 1, 4, 3, 1, 3, 2, 2, 3, 4, 1, 0, 3, 4, 0,
                    3, 5, 2, 1, 4, 5, 0, 2, 0, 2, 2, 2, 0, 4, 2, 2], device='cuda:0')
            '''

            result = prediction == target
            # print("result:")
            # print(result)
            '''
            tensor([False, False,  True, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False,  True, False,
                     True,  True, False, False, False, False, False, False, False, False,
                    False, False, False, False,  True, False, False, False, False,  True,
                     True, False,  True,  True, False, False, False, False,  True, False,
                     True, False, False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False,  True, False,
                    False, False,  True, False, False, False, False, False, False, False,
                    False,  True, False, False, False, False, False,  True, False, False,
                    False, False, False, False, False, False, False, False,  True, False,
                    False, False, False, False,  True, False,  True, False,  True, False,
                    False, False,  True, False,  True, False, False, False, False, False,
                    False, False, False, False, False, False, False, False, False,  True,
                    False, False, False, False, False, False, False, False,  True, False,
                    False, False, False,  True, False, False,  True, False, False, False,
                    False, False, False, False, False, False,  True, False, False, False,
                     True, False, False,  True, False, False, False, False,  True, False,
                    False, False, False, False,  True, False,  True, False, False, False,
                    False, False,  True, False,  True, False, False,  True,  True, False,
                    False, False, False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False,  True, False, False,
                     True, False, False,  True, False, False,  True, False, False, False,
                     True, False, False, False, False, False,  True, False, False,  True,
                    False, False, False, False, False,  True, False, False, False, False,
                    False, False, False,  True, False, False, False, False, False, False,
                    False, False, False, False, False, False], device='cuda:0')
            '''

            for target_idx in range(len(target)):
                target_single = target[target_idx]
                correct[target_single] += result[target_idx].item()
                total[target_single] += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        acc_str = 'Accuracy: %f' % (sum(correct) / sum(total))
        for acc_idx in range(len(label)):
            try:
                acc = correct[acc_idx] / total[acc_idx]
            except:
                acc = 0
            finally:
                acc_str += '\tclass:%s\tacc:%f\t' % (trans_list[acc_idx], acc)

        print(acc_str)

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, acc_str


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.arch + '_' + args.action)

    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def report(output, target):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(6, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []

        correct_0 = correct[0].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_0.mul_(100.0 / batch_size))

        correct_1 = correct[1].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_1.mul_(100.0 / batch_size))

        correct_2 = correct[2].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_2.mul_(100.0 / batch_size))

        correct_3 = correct[3].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_3.mul_(100.0 / batch_size))

        correct_4 = correct[4].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_4.mul_(100.0 / batch_size))

        correct_5 = correct[5].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_5.mul_(100.0 / batch_size))

        print(res)

        return res


def data_save(root, file):
    if not os.path.exists(root):
        # os.mknod(root)
        open(root, 'wb')
    file_temp = open(root, 'r')
    lines = file_temp.readlines()
    if not lines:
        epoch = -1
    else:
        epoch = lines[-1][:lines[-1].index(' ')]
    epoch = int(epoch)
    file_temp.close()
    file_temp = open(root, 'a')
    for line in file:
        if line > epoch:
            file_temp.write(str(line) + " " + str(file[line]) + '\n')
    file_temp.close()


if __name__ == '__main__':
    main()
