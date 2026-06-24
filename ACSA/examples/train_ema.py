# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import warnings
from bisect import bisect_right

warnings.filterwarnings('ignore')
import os
import sys
os.chdir(sys.path[0])
sys.path.append("../")
import os.path as osp
import random
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from lib import datasets
from lib import models
from lib.models.cm import ClusterMemory, ClusterMemory_ema
from lib.trainers import ClusterContrastTrainer, ClusterContrastTrainer_ema
from lib.evaluators import Evaluator, extract_features, extract_features_mask
from lib.utils.data import IterLoader
from lib.utils.data import transforms as T
from lib.utils.data.sampler import RandomMultipleGallerySampler
from lib.utils.data.preprocessor import Preprocessor_mutual
from lib.utils.logging import Logger
from lib.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from lib.utils.faiss_rerank import compute_jaccard_distance
from lib.utils.caj_rerank import compute_jaccard_distance_caj
start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = '/data/tx/datasets'
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):
    if args.self_norm:
        normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    else:
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor_mutual(train_set, root=dataset.images_dir, transform=train_transformer, mutual=False),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader



def get_test_loader(args, dataset, height, width, batch_size, workers, testset=None):
    if args.self_norm:
        normalizer = T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    else:
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor_mutual(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)



    return test_loader


def create_model(args):
    if 'resnet' in args.arch:
        model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                num_classes=0, pooling_type=args.pooling_type,pretrained_path=args.pretrained_path)

    else:
        model = models.create(args.arch,img_size=(args.height,args.width),drop_path_rate=args.drop_path_rate
                , pretrained_path = args.pretrained_path,hw_ratio=args.hw_ratio, conv_stem=args.conv_stem)

    model.cuda()
    model = nn.DataParallel(model)
    return model

def create_model_stage2(args):
    if 'resnet' in args.arch:
        model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                num_classes=0, pooling_type=args.pooling_type,pretrained_path=args.pretrained_path)
        model_ema = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                num_classes=0, pooling_type=args.pooling_type,pretrained_path=args.pretrained_path)
    else:
        model = models.create(args.arch,img_size=(args.height,args.width),drop_path_rate=args.drop_path_rate
                , pretrained_path = args.pretrained_path,hw_ratio=args.hw_ratio, conv_stem=args.conv_stem)
        model_ema = models.create(args.arch, img_size=(args.height, args.width), drop_path_rate=args.drop_path_rate
                              , pretrained_path=args.pretrained_path, hw_ratio=args.hw_ratio, conv_stem=args.conv_stem)
    # use CUDA

    model.cuda()
    model_ema.cuda()
    model = nn.DataParallel(model)
    model_ema = nn.DataParallel(model_ema)
    return model, model_ema

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    cudnn.benchmark = False

    print("MCL Loss weight:{}".format(args.loss_weight))
    print("MSE Loss:{}".format(args.loss_weight1))
    print("Mask ratio:{}".format(args.mask_ratio))

    if args.stage1:
        main_worker_stage1(args)
    if args.stage2:
        main_worker_stage2(args)

def main_worker_stage1(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = False

    sys.stdout = Logger(osp.join(args.logs_dir, 'log_stage1.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(args, dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    print('optimizer: %s'%(args.optimizer))
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)


    # generate camera labels
    if args.dataset == 'msmt17':
        cam_labels = np.array([cid - 1 for _, _, cid in sorted(dataset.train)])
    else:
        cam_labels = np.array([cid for _, _, cid in sorted(dataset.train)])

    # Trainer
    trainer = ClusterContrastTrainer(args,model)

    for epoch in range(args.epochs):
        print('=> EPOCH num={}'.format(epoch))
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            time.sleep(0.5)
            cluster_loader = get_test_loader(args, dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))
            print('=> Extract features...')
            features, _ = extract_features(model, cluster_loader, print_freq=50)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)

            print('=> Extract mask features...')
            features_mask, _ = extract_features_mask(model, cluster_loader, print_freq=50)
            features_mask = torch.cat([features_mask[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)

            if args.caj:
                print(" Using caj jaccard_distance! ")
                rerank_dist = compute_jaccard_distance_caj(features, cam_labels=cam_labels, epoch=epoch, args=args)
            else:
                rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)

            if epoch == 0:
                # DBSCAN cluster
                eps = args.eps
                print('Clustering criterion: eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

            # print("epoch: {} \n pseudo_labels: {}".format(epoch, pseudo_labels.tolist()[:100]))

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        @torch.no_grad()
        def generate_cluster_features_random(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                random.choice(torch.stack(centers[idx], dim=0)) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        def generate_random_features(labels, features, num_cluster, num_instances):
            indexes = np.zeros(num_cluster*num_instances)
            for i in range(num_cluster):
                index = [i+k*num_cluster for k in range(num_instances)]
                samples = np.random.choice(np.where(pseudo_labels==i)[0], num_instances, True)
                indexes[index] = samples
            memory_features = features[indexes]
            return memory_features

        cluster_features = generate_cluster_features(pseudo_labels, features)

        if args.dataset == 'market1501':
            cluster_features_random = generate_random_features(pseudo_labels, features, num_cluster, 4)
        else:
            cluster_features_random = generate_cluster_features_random(pseudo_labels, features)

        cluster_features_mask = generate_cluster_features(pseudo_labels, features_mask)
        del cluster_loader, features

        # Create hybrid memory
        memory = ClusterMemory(args, model.module.num_features, num_cluster, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard, loss_weight=args.loss_weight, loss_weight1=args.loss_weight1).cuda()
        memory.features = F.normalize(cluster_features, dim=1).cuda()
        memory.features_random = F.normalize(cluster_features_random, dim=1).cuda()
        memory.features_mask = F.normalize(cluster_features_mask, dim=1).cuda()

        trainer.memory = memory

        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))

        print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset)

        curr_lr = optimizer.param_groups[0]['lr']
        print('=> Current Lr: {:.2e}'.format(curr_lr))
        time.sleep(0.5)
        train_loader.new_epoch()
        time.sleep(0.5)
        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader),mask_ratio = args.mask_ratio,con_weight=args.con_weight
                      )

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            _,mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir + '/log_s1/', 'model.pth.tar'))


            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    end_time = time.monotonic()
    print('Stage1 Total running time: ', timedelta(seconds=end_time - start_time))

def main_worker_stage2(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    sys.stdout = Logger(osp.join(args.logs_dir, 'log_stage2.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(args, dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model, model_ema = create_model_stage2(args)
    ckpt_path = osp.join(args.logs_dir, 'log_s1', 'model_best.pth.tar')
    if osp.exists(ckpt_path):
        checkpoint = load_checkpoint(ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])
        model_ema.load_state_dict(checkpoint['state_dict'])
        print(f"=> Loaded checkpoint from {ckpt_path}")
    else:
        print(f"=> No checkpoint found at {ckpt_path}, skipping weight loading.")

    # Evaluator
    evaluator = Evaluator(model)
    evaluator_ema = Evaluator(model_ema)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    print('optimizer: %s'%(args.optimizer))
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # generate camera labels
    if args.dataset == 'msmt17':
        cam_labels = np.array([cid - 1 for _, _, cid in sorted(dataset.train)])
    else:
        cam_labels = np.array([cid for _, _, cid in sorted(dataset.train)])

    # Trainer
    trainer = ClusterContrastTrainer_ema(args,model,model_ema)

    for epoch in range(args.epochs1):
        print('=> EPOCH num={}'.format(epoch))
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            time.sleep(0.5)
            cluster_loader = get_test_loader(args, dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))
            print('=> Extract features...')
            features, _ = extract_features(model, cluster_loader, print_freq=50)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)

            print('=> Extract EMA Model features...')
            features_ema, _ = extract_features(model_ema, cluster_loader, print_freq=50)
            features_ema = torch.cat([features_ema[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)

            print('=> Extract mask features...')
            features_mask, _ = extract_features_mask(model, cluster_loader, print_freq=50)
            features_mask = torch.cat([features_mask[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)

            if args.caj:
                print(" Using caj jaccard_distance! ")
                rerank_dist = compute_jaccard_distance_caj(features, cam_labels=cam_labels, epoch=epoch, args=args)
            else:
                rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)

            if epoch == 0:
                # DBSCAN cluster
                eps = args.eps
                print('Clustering criterion: eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

            # print("epoch: {} \n pseudo_labels: {}".format(epoch, pseudo_labels.tolist()[:100]))

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        @torch.no_grad()
        def generate_cluster_features_random(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                random.choice(torch.stack(centers[idx], dim=0)) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        def generate_random_features(labels, features, num_cluster, num_instances):
            indexes = np.zeros(num_cluster*num_instances)
            for i in range(num_cluster):
                index = [i+k*num_cluster for k in range(num_instances)]
                samples = np.random.choice(np.where(pseudo_labels==i)[0], num_instances, True)
                indexes[index] = samples
            memory_features = features[indexes]
            return memory_features

        cluster_features = generate_cluster_features(pseudo_labels, features)


        if args.dataset == 'market1501':
            fixk = args.k
            cluster_features_random = generate_random_features(pseudo_labels, features_ema, num_cluster, fixk)
            print(" market1501 generate_random_features k : {}".format(fixk))
        else:
            fixk = args.k
            cluster_features_random = generate_random_features(pseudo_labels, features_ema, num_cluster, fixk)
            print(" msmt17 generate_random_features k : {}".format(fixk))

        cluster_features_mask = generate_cluster_features(pseudo_labels, features_mask)

        del cluster_loader, features

        # Create hybrid memory
        memory = ClusterMemory_ema(args, model.module.num_features, num_cluster, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard, loss_weight=args.loss_weight, loss_weight1=args.loss_weight1).cuda()
        memory.features = F.normalize(cluster_features, dim=1).cuda()
        memory.features_random = F.normalize(cluster_features_random, dim=1).cuda()
        memory.features_mask = F.normalize(cluster_features_mask, dim=1).cuda()

        trainer.memory = memory

        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))

        print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset)

        curr_lr = optimizer.param_groups[0]['lr']
        print('=> Current Lr: {:.2e}'.format(curr_lr))
        time.sleep(0.5)
        train_loader.new_epoch()
        time.sleep(0.5)
        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader),mask_ratio = args.mask_ratio,con_weight=args.con_weight
                      )

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP_1 = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            mAP_2 = evaluator_ema.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mAP_1 > best_mAP) or (mAP_2 > best_mAP)
            best_mAP = max(mAP_1, mAP_2, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir+ '/log_s2/', 'model.pth.tar'))
            save_checkpoint({
                'state_dict': model_ema.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, (is_best and (mAP_1 <= mAP_2)), fpath=osp.join(args.logs_dir+ '/log_s2/', 'model_ema.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  model_ema mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP_1, mAP_2, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir+ '/log_s2/', 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=8,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='vit_small',
                        choices=models.names())
    parser.add_argument('-pp', '--pretrained-path', type=str, default='')
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    #vit
    parser.add_argument('--drop-path-rate', type=float, default=0.1)
    parser.add_argument('--hw-ratio', type=int, default=2)
    parser.add_argument('--self-norm', action="store_true")
    parser.add_argument('--conv-stem', action="store_true")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--epochs1', type=int, default=50)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")

    parser.add_argument('--loss_weight', type=float, default=0.04)
    # label_preserving (LP) loss parameters mask_ratio
    parser.add_argument('--loss_weight1', type=float, default=0.5)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--stage1', action="store_true")
    parser.add_argument('--stage2', action="store_true")
    parser.add_argument('--resume', action="store_true")



    parser.add_argument('--caj', action='store_true')
    #CKRNNs
    parser.add_argument('--ckrnns', action='store_true')
    parser.add_argument('--k1-intra', type=int, default=5)
    parser.add_argument('--k1-inter', type=int, default=20)

    # CLQE
    parser.add_argument('--clqe', action='store_true')
    parser.add_argument('--k2-intra', type=int, default=2)
    parser.add_argument('--k2-inter', type=int, default=4)

    main()
