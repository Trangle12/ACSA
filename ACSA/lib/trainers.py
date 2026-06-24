from __future__ import print_function, absolute_import
import time
import torch
from torch.cuda import amp
from .utils.meters import AverageMeter

class ClusterContrastTrainer(object):
    def __init__(self, args,encoder, memory=None, memory_mask=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.memory_mask = memory_mask

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, fp16=True,  mask_ratio = 0.75):
        print("use mask_ratio:{}".format(mask_ratio))
        self.encoder.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        # amp fp16 training
        scaler = amp.GradScaler() if fp16 else None
        for i in range(train_iters):
            # 1) load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)
            imgs, labels, indexes = self._parse_data(inputs)
            with amp.autocast(enabled=fp16):

                f_out = self._forward(imgs, mask=False, mask_ratio=mask_ratio)

                f_out_mask  = self._forward(imgs, mask=True, mask_ratio=mask_ratio)

                loss = self.memory(f_out, f_out_mask, labels,epoch)

            optimizer.zero_grad()

            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            losses.update(loss.item())
            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs, mask, mask_ratio):
        return self.encoder(inputs, mask, mask_ratio)

class ClusterContrastTrainer_ema(object):
    def __init__(self, args,encoder, encoder_ema, memory=None, memory_mask=None, alpha=0.999):
        super(ClusterContrastTrainer_ema, self).__init__()
        self.encoder = encoder
        self.encoder_ema = encoder_ema
        self.memory = memory
        self.memory_mask = memory_mask
        self.alpha = alpha

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, fp16=True, mask_ratio = 0.75):
        self.encoder.train()
        self.encoder_ema.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        # amp fp16 training
        scaler = amp.GradScaler() if fp16 else None
        for i in range(train_iters):
            # 1) load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)
            imgs, labels, indexes = self._parse_data(inputs)
            with amp.autocast(enabled=fp16):

                f_out = self._forward(imgs, mask=False, mask_ratio=mask_ratio)

                f_out_mask  = self._forward(imgs, mask=True, mask_ratio=mask_ratio)

                loss = self.memory(f_out, f_out_mask,  labels)

            optimizer.zero_grad()

            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            self._update_ema_variables(self.encoder, self.encoder_ema, 0.999)

            losses.update(loss.item())
            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs, mask, mask_ratio):
        return self.encoder(inputs, mask, mask_ratio)

    def _forward_ema(self, inputs, mask, mask_ratio):
        return self.encoder_ema(inputs, mask, mask_ratio)
    def _update_ema_variables(self, model, ema_model, alpha):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)