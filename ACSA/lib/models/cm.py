import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.cuda import amp


class CM(autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)


        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))



class ClusterMemory(nn.Module, ABC):
    def __init__(self, args,num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False, loss_weight=0.04, loss_weight1 = 0.5):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.dataset = args.dataset
        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.loss_weight = loss_weight

    def forward(self, inputs, inputs_mask, targets, epoch):
        inputs = F.normalize(inputs, dim=1).cuda()
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)

        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)
        inputs_mask = F.normalize(inputs_mask, dim=1).cuda()
        outputs_mask = cm(inputs_mask, targets, self.features_mask, self.momentum)

        outputs_mask /= self.temp
        loss1 = F.cross_entropy(outputs_mask, targets)

        outputs_mask_mse = inputs_mask.mm(self.features.t()) / self.temp
        loss_mse = softmax_mse_loss(outputs_mask_mse.t().contiguous(), outputs.t().contiguous())

        outputs1 = inputs.mm(self.features_random.t()) / self.temp
        outputs_mask_mse1 = inputs_mask.mm(self.features_random.t()) / self.temp
        loss_mse1 = softmax_mse_loss(outputs_mask_mse1.t().contiguous(), outputs1.t().contiguous())

        return loss + self.loss_weight1 * loss_mse + (1 - self.loss_weight1) * loss_mse1 + self.loss_weight * loss1

class ClusterMemory_ema(nn.Module, ABC):
    def __init__(self, args,num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False, loss_weight=0.04, loss_weight1 = 0.5):
        super(ClusterMemory_ema, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.dataset = args.dataset
        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.loss_weight = loss_weight

    def forward(self, inputs, inputs_mask, targets):
        inputs = F.normalize(inputs, dim=1).cuda()
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)
        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)

        inputs_mask = F.normalize(inputs_mask, dim=1).cuda()
        outputs_mask = cm(inputs_mask, targets, self.features_mask, self.momentum)
        outputs_mask /= self.temp
        loss1 = F.cross_entropy(outputs_mask, targets)

        outputs_mask_mse = inputs_mask.mm(self.features.t()) / self.temp
        loss_mse = softmax_mse_loss(outputs_mask_mse.t().contiguous(), outputs.t().contiguous())

        outputs1 = inputs.mm(self.features_random.t()) / self.temp
        outputs_mask_mse1 = inputs_mask.mm(self.features_random.t()) / self.temp
        loss_mse1 = softmax_mse_loss(outputs_mask_mse1.t().contiguous(), outputs1.t().contiguous())

        return loss + self.loss_weight1 * loss_mse + (1 - self.loss_weight1) * loss_mse1 + self.loss_weight * loss1


def softmax_mse_loss(input_logits, target_logits):
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes
