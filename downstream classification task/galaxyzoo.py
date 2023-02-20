import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import PIL
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
# import tensorflow as tf
import torchvision.datasets as datasets
import torchvision.models as models
import copy
import moco.builder
# In  pre-training  folder (builder.py)

import os
from torch.autograd import Variable

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)

            res.append(correct_k.mul_(100.0 / batch_size))

        return res

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 1))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 60 == 0:
            progress.display(i)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    TP, FP, FN, TN = 0, 0, 0, 0
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            acc1, acc5 = accuracy(output, target, topk=(1, 1))
            loss = criterion(output, target)

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            outputs = model(images)
            # print(outputs)
            predict_y = torch.max(outputs, dim=1)[1]
            # print(predict_y)
            # print(target)
            labels = [0, 1, 2, 3, 4]

            for label in labels:
                preds_tmp = np.array([1 if pred == label else 0 for pred in predict_y])
                trues_tmp = np.array([1 if true == label else 0 for true in target])

            TP += ((preds_tmp == 1) & (trues_tmp == 1)).sum()
            TN += ((preds_tmp == 0) & (trues_tmp == 0)).sum()
            FN += ((preds_tmp == 0) & (trues_tmp == 1)).sum()
            FP += ((preds_tmp == 1) & (trues_tmp == 0)).sum()

            # precision = TP / (TP + FP)
            # recall = TP / (TP + FN)
            # f1 = 2 * precision * recall / (precision + recall)
            # print('precision: %.3f  recall: %.3f  f1: %.3f' %(precision, recall, f1))

            if i % 180 == 0:
                progress.display(i)
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * precision * recall / (precision + recall)
                print('precision: %.3f  recall: %.3f  f1: %.3f' % (precision, recall, f1))

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} '
              .format(top1=top1))

    return top1.avg

# ---------------------------------------------------------------------------
# create model
arch = 'resnet50'
print("=> creating model '{}'".format(arch))
model = models.__dict__[arch](num_classes=5)

# freeze all layers but the last fc
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False

print("#######################################################################")

# freeze all layers

# init the fc layer
model.fc.weight.data.normal_(mean=0.0, std=0.01)
model.fc.bias.data.zero_()

# ------------------------  Loading the pre-trained module (start) ------------------------------
# print(model)
pretrained_path = 'checkpoint_0800.pth.tar'
print("=> loading checkpoint '{}'".format(pretrained_path))
checkpoint = torch.load(pretrained_path)
state_dict = checkpoint['state_dict']

queue_state_dict = copy.deepcopy(state_dict)

for k in list(state_dict.keys()):
    # retain only encoder_q up to before the embedding layer
    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
        # remove prefix
        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
    # delete renamed or unused k
    del state_dict[k]

for k in list(queue_state_dict.keys()):
    # retain only encoder_q up to before the embedding layer
    if k.startswith('module.'):
        # remove prefix
        queue_state_dict[k[len("module."):]] = queue_state_dict[k]
    # delete renamed or unused k
    del queue_state_dict[k]

# ------------------------  Loading the pre-trained module (finished) ------------------------------

msg = model.load_state_dict(state_dict, strict=False)

assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

print("=> loaded pre-trained model '{}'".format(pretrained_path))

model.cuda()

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()

# optimize only the linear classifier
parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

# assert len(parameters) == 2  # fc.weight, fc.bias

schedule = [60, 80]
start_epoch = 0
# lr = 0.0015
momentum = 0.9
weight_decay = 0.0

# optimizer = torch.optim.SGD(parameters, lr=0.015, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.Adam(parameters,lr=0.001,betas=(0.9, 0.999),eps=1e-08,weight_decay=0,amsgrad=False)  #88.5
# optimizer = torch.optim.RMSprop(parameters,lr=0.01,alpha=0.99,eps=1e-08,weight_decay=0,momentum=0,centered=False) #87


cudnn.benchmark = True
traindir = 'fenge_train'
valdir = 'fenge_test'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        #transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

train_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
       #transforms.Resize(256),
       #transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
        batch_size=32, shuffle=True,
    pin_memory=True)

# def adjust_learning_rate(optimizer, epoch):
#     lr = 0.05
#     for milestone in schedule:
#         lr *= 0.1 if epoch >= milestone else 1.
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     print(lr)

best_acc1 = 0
model.cuda()

for epoch in range(0, 50):
    # adjust_learning_rate(optimizer, epoch)
    train(train_loader, model, criterion, optimizer, epoch)
    acc1 = validate(val_loader, model, criterion)
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)
    print(best_acc1)