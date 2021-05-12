from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import  filterfalse as ifilterfalse

#######################################################
# 0. Main loss functions
#######################################################

class JaccardLoss(nn.Module):
    """Jaccard loss.
    """

    # binary case

    def __init__(self, size_average=True, reduce=True, smooth=1.0):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth
        self.reduce = reduce

    def jaccard_loss(self, pred, target):
        loss = 0.
        # for each sample in the batch
        for index in range(pred.size()[0]):
            iflat = pred[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()
            loss += 1 - ((intersection + self.smooth) /
                         (iflat.sum() + tflat.sum() - intersection + self.smooth))
            # print('loss:',intersection, iflat.sum(), tflat.sum())

        # size_average=True for the jaccard loss
        return loss / float(pred.size()[0])

    def jaccard_loss_batch(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        loss = 1 - ((intersection + self.smooth) /
                    (iflat.sum() + tflat.sum() - intersection + self.smooth))
        # print('loss:',intersection, iflat.sum(), tflat.sum())
        return loss

    def forward(self, pred, target):
        # _assert_no_grad(target)
        if not (target.size() == pred.size()):
            raise ValueError("Target size ({}) must be the same as pred size ({})".format(target.size(), pred.size()))
        if self.reduce:
            loss = self.jaccard_loss(pred, target)
        else:
            loss = self.jaccard_loss_batch(pred, target)
        return loss


# def make_one_hot(input, num_classes):
#     """Convert class index tensor to one hot encoding tensor.
#     Args:
#          input: A tensor of shape [N, 1, *]
#          num_classes: An int of number of class
#     Returns:
#         A tensor of shape [N, num_classes, *]
#     """
#     shape = np.array(input.shape)
#     shape[1] = num_classes
#     shape = tuple(shape)
#     result = torch.zeros(shape)
#     result = result.scatter_(1, input.cpu(), 1)
#
#     return result
#
#
# class BinaryDiceLoss(nn.Module):
#     """Dice loss of binary class
#     Args:
#         smooth: A float number to smooth loss, and avoid NaN error, default: 1
#         p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
#         predict: A tensor of shape [N, *]
#         target: A tensor of shape same with predict
#         reduction: Reduction method to apply, return mean over batch if 'mean',
#             return sum if 'sum', return a tensor of shape [N,] if 'none'
#     Returns:
#         Loss tensor according to arg reduction
#     Raise:
#         Exception if unexpected reduction
#     """
#     def __init__(self, smooth=1e-6, p=2, reduction='mean'):
#         super(BinaryDiceLoss, self).__init__()
#         self.smooth = smooth
#         self.p = p
#         self.reduction = reduction
#
#     def forward(self, predict, target):
#         predict = predict.cuda()
#         target = target.cuda()
#         assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
#         predict = predict.contiguous().view(predict.shape[0], -1)
#         target = target.contiguous().view(target.shape[0], -1)
#
#         num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
#         den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
#
#         loss = 1 - num / den
#
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         elif self.reduction == 'none':
#             return loss
#         else:
#             raise Exception('Unexpected reduction {}'.format(self.reduction))
#
#
# class DiceLoss(nn.Module):
#     """Dice loss, need one hot encode input
#     Args:
#         weight: An array of shape [num_classes,]
#         ignore_index: class index to ignore
#         predict: A tensor of shape [N, C, *]
#         target: A tensor of same shape with predict
#         other args pass to BinaryDiceLoss
#     Return:
#         same as BinaryDiceLoss
#     """
#     def __init__(self, weight=None, ignore_index=None, **kwargs):
#         super(DiceLoss, self).__init__()
#         self.kwargs = kwargs
#
#         self.weight = None
#         self.ignore_index = ignore_index
#
#     def forward(self, predict, target, *args):
#         # self.weight = frequency(target)
#         target = target.long()
#         predict = torch.cat([1-predict, predict], dim=1)
#         target = make_one_hot(target, 2)
#
#         assert predict.shape == target.shape, 'predict & target shape do not match'
#         dice = BinaryDiceLoss(**self.kwargs)
#         total_loss = 0
#         predict = F.softmax(predict, dim=1)
#
#         for i in range(target.shape[1]):
#             if i != self.ignore_index:
#                 # one-hot 第i个class 666
#                 dice_loss = dice(predict[:, i], target[:, i])
#
#                 #print(self.weight)
#                 if self.weight is not None:
#                     assert self.weight.shape[0] == target.shape[1], \
#                         'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
#                     dice_loss *= self.weight[i]
#                 total_loss += dice_loss
#         loss = total_loss/target.shape[1]
#         # print(loss)
#         return loss


class DiceLoss(nn.Module):
    """DICE loss.
    """

    # https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/

    def __init__(self, size_average=True, reduce=True, smooth=100.0, power=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduce = reduce
        self.power = power

    def dice_loss(self, pred, target):
        loss = 0.

        for index in range(pred.size()[0]):
            iflat = pred[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()
            if self.power == 1:
                loss += 1 - ((2. * intersection + self.smooth) /
                             (iflat.sum() + tflat.sum() + self.smooth))
            else:
                loss += 1 - ((2. * intersection + self.smooth) /
                             ((iflat ** self.power).sum() + (tflat ** self.power).sum() + self.smooth))

        # size_average=True for the dice loss
        return loss / float(pred.size()[0])

    def dice_loss_batch(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        if self.power == 1:
            loss = 1 - ((2. * intersection + self.smooth) /
                        (iflat.sum() + tflat.sum() + self.smooth))
        else:
            loss = 1 - ((2. * intersection + self.smooth) /
                        ((iflat ** self.power).sum() + (tflat ** self.power).sum() + self.smooth))
        return loss

    def forward(self, pred, target, *args):
        # _assert_no_grad(target)
        if not (target.size() == pred.size()):
            raise ValueError("Target size ({}) must be the same as pred size ({})".format(target.size(), pred.size()))

        if self.reduce:
            loss = self.dice_loss(pred, target)
        else:
            loss = self.dice_loss_batch(pred, target)
        return loss


class WeightedMSE(nn.Module):
    """Weighted mean-squared error.
    """

    def __init__(self):
        super().__init__()

    def weighted_mse_loss(self, pred, target, weight):
        s1 = torch.prod(torch.tensor(pred.size()[2:]).float())
        s2 = pred.size()[0]
        norm_term = (s1 * s2).cuda()
        if weight is None:
            return torch.sum((pred - target) ** 2) / norm_term
        else:
            return torch.sum(weight * (pred - target) ** 2) / norm_term

    def forward(self, pred, target, weight=None):
        # _assert_no_grad(target)
        return self.weighted_mse_loss(pred, target, weight)


class WeightedBCE(nn.Module):
    """Weighted binary cross-entropy.
    """

    def __init__(self, label_smooth=0.01, n_class=2, size_average=True, reduce=True):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.label_smooth = label_smooth
        self.n_class = n_class

    def forward(self, pred, target, weight=None):
        # _assert_no_grad(target)
        # weight torch.Size([4, 1, 32, 256, 256]) pixel-wise weight
        # pred = torch.clamp(pred, self.smooth, 1.0 - self.smooth)
        # target[target == 0] = self.label_smooth/(self.n_class-1)
        # target[target == 1] = 1-self.label_smooth
        # target =target.detach()
        #
        # one_hot_label = torch.zeros_like(pred)
        # one_hot_label = one_hot_label.scatter(1, target.long(), 1)
        # one_hot_label = one_hot_label * (1 - self.label_smooth) + (1 - one_hot_label) * self.label_smooth / (self.n_class - 1)
        # one_hot_label = one_hot_label[:, 1, ...].squeeze(1)
        return F.binary_cross_entropy(pred, target, weight)


class WeightedCE(nn.Module):
    """Mask weighted multi-class cross-entropy (CE) loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight_mask=[0.07, 0.93]):
        # Different from, F.binary_cross_entropy, the "weight" parameter
        # in F.cross_entropy is a manual rescaling weight given to each
        # class. Therefore we need to multiply the weight mask after the
        # loss calculation.
        weight_mask = torch.tensor([0.07, 0.93]).cuda()
        loss = F.cross_entropy(pred, target, reduction='none')
        if weight_mask is not None:
            loss = loss * weight_mask
        return loss.mean()


#######################################################
# 1. Regularization
#######################################################

class BinaryReg(nn.Module):
    """Regularization for encouraging the outputs to be binary.
    """

    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred):
        diff = pred - 0.5
        diff = torch.clamp(torch.abs(diff), min=1e-2)
        loss = (1.0 / diff).mean()
        return self.alpha * loss


# class OhemCrossEntropy(nn.Module):
#     def __init__(self, ignore_label=-1, thres=0.9,
#                  min_kept=131072, weight=None):
#         super(OhemCrossEntropy, self).__init__()
#         self.thresh = thres
#         self.min_kept = max(1, min_kept)
#         self.ignore_label = ignore_label
#         self.criterion = WeightedBCE()
#         # self.criterion = nn.CrossEntropyLoss(weight=weight,
#         #                                      ignore_index=ignore_label,
#         #                                      reduction='none')
#
#     def forward(self, pred, target, weight, *kwargs):
#         # ph, pw = pred.size(2), pred.size(3)
#         # h, w = target.size(1), target.size(2)
#         # if ph != h or pw != w:
#         #     score = F.upsample(input=score, size=(h, w), mode='bilinear')
#         # pred = F.softmax(score, dim=1)
#         pixel_losses = self.criterion(pred, target, weight=weight).contiguous().view(-1)
#         # mask = target.contiguous().view(-1) != self.ignore_label
#         #
#         # mask, pred, target, weight = mask.cpu(), pred.cpu(), target.cpu(), weight.cpu()
#         tmp_target = target.clone().long()
#         tmp_target[tmp_target == self.ignore_label] = 0
#         pred = pred.gather(1, tmp_target) # pred.gather(1, tmp_target.unsqueeze(1))
#         pred, ind = pred.contiguous().view(-1).sort()
#         min_value = pred[min(self.min_kept, pred.numel() - 1)]
#         threshold = max(min_value, self.thresh)
#
#         pixel_losses = pixel_losses[ind]
#         pixel_losses = pixel_losses[pred < threshold]
#         return pixel_losses.mean()

# reference: https://zhuanlan.zhihu.com/p/80594704

class Py_sigmoid_focal_loss(nn.Module):
    def __init__(self, gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):

        super(Py_sigmoid_focal_loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.avg_factor = avg_factor

    def forward(self, pred, target, weight, *kwargs):
        # pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred) * target + pred * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) *
                        (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, weight=weight, reduction='none') * focal_weight

        # Online Hard Example Mining: top x% losses (pixel-wise). Refer to http://www.robots.ox.ac.uk/~tvg/publications/2017/0026.pdf
        OHEM_percent = 0.9
        OHEM, _ = loss.topk(k=int(OHEM_percent * [*loss.shape][0]))


        return OHEM.mean()



from torch.autograd import Variable



# lovasz softmax loss function
# reference: https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/demo_binary.ipynb
# if I only use lovasz softmax loss, the training stage cannot converge.
class Lovasz_softmax(torch.nn.modules.Module):
    def __init__(self):
        super(Lovasz_softmax, self).__init__()


    #  --------------------------- BINARY LOSSES ---------------------------
    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard


    def iou_binary(self, preds, labels, EMPTY=1., ignore=None, per_image=True):
        """
        IoU for foreground class
        binary: 1 foreground, 0 background
        """
        if not per_image:
            preds, labels = (preds,), (labels,)
        ious = []
        for pred, label in zip(preds, labels):
            intersection = ((label == 1) & (pred == 1)).sum()
            union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
            if not union:
                iou = EMPTY
            else:
                iou = float(intersection) / float(union)
            ious.append(iou)
        iou = self.mean(ious)  # mean accross images if per_image
        return 100 * iou


    def iou(self, preds, labels, C, EMPTY=1., ignore=None, per_image=False):
        """
        Array of IoU for each (non ignored) class
        """
        if not per_image:
            preds, labels = (preds,), (labels,)
        ious = []
        for pred, label in zip(preds, labels):
            iou = []
            for i in range(C):
                if i != ignore:  # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                    intersection = ((label == i) & (pred == i)).sum()
                    union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                    if not union:
                        iou.append(EMPTY)
                    else:
                        iou.append(float(intersection) / float(union))
            ious.append(iou)
        ious = [self.mean(iou) for iou in zip(*ious)]  # mean accross images if per_image
        return 100 * np.array(ious)


    # --------------------------- BINARY LOSSES ---------------------------


    def lovasz_hinge(self, logits, labels, per_image=True, ignore=None):
        """
        Binary Lovasz hinge loss
          logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
          labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
          per_image: compute the loss per image instead of per batch
          ignore: void class id
        """
        if per_image:
            loss = self.mean(self.lovasz_hinge_flat(*self.flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                        for log, lab in zip(logits, labels))
        else:
            loss = self.lovasz_hinge_flat(*self.flatten_binary_scores(logits, labels, ignore))
        return loss


    def lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
          ignore: label to ignore
        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * Variable(signs))
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), Variable(grad))
        return loss


    def flatten_binary_scores(self, scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = scores.view(-1)
        labels = labels.view(-1)
        if ignore is None:
            return scores, labels
        valid = (labels != ignore)
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels


    # class StableBCELoss(self, torch.nn.modules.Module):
    #     def __init__(self):
    #         super(StableBCELoss, self).__init__()
    #
    #     def forward(self, input, target):
    #         neg_abs = - input.abs()
    #         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    #         return loss.mean()
    #

    def binary_xloss(self, logits, labels, ignore=None):
        """
        Binary Cross entropy loss
          logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
          labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
          ignore: void class id
        """
        logits, labels = self.flatten_binary_scores(logits, labels, ignore)
        loss = self.StableBCELoss()(logits, Variable(labels.float()))
        return loss


    # --------------------------- MULTICLASS LOSSES ---------------------------


    def lovasz_softmax(self, probas, labels, classes='present', per_image=False, ignore=None):
        """
        Multi-class Lovasz-Softmax loss
          probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                  Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
          labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
          per_image: compute the loss per image instead of per batch
          ignore: void class labels
        """
        if per_image:
            loss = self.mean(self.lovasz_softmax_flat(*self.flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                        for prob, lab in zip(probas, labels))
        else:
            loss = self.lovasz_softmax_flat(*self.flatten_probas(probas, labels, ignore), classes=classes)
        return loss


    def lovasz_softmax_flat(self, probas, labels, classes='present'):
        """
        Multi-class Lovasz-Softmax loss
          probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [P] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = (labels == c).float()  # foreground for class c
            if (classes is 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (Variable(fg) - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, Variable(self.lovasz_grad(fg_sorted))))
        return self.mean(losses)


    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W)
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = (labels != ignore)
        vprobas = probas[valid.nonzero().squeeze()]
        vlabels = labels[valid]
        return vprobas, vlabels


    def xloss(self, logits, labels, ignore=None):
        """
        Cross entropy loss
        """
        return F.cross_entropy(logits, Variable(labels), ignore_index=255)


    # --------------------------- HELPER FUNCTIONS ---------------------------
    def isnan(self, x):
        return x != x


    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """
        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(self.isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n

    def forward(self, input, target, weight, per_image=True, ignore=None):
        """
        Modified by function lovasz_hinge.
        Binary Lovasz hinge loss
          logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
          labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
          per_image: compute the loss per image instead of per batch
          ignore: void class id
        """
        if per_image:
            loss = self.mean(self.lovasz_hinge_flat(*self.flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                        for log, lab in zip(input, target))
        else:
            loss = self.lovasz_hinge_flat(*self.flatten_binary_scores(input, target, ignore))
        return loss



# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0.25, alpha=2, smooth=1e-6, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha, (float, int, long)): self.alpha = torch.Tensor([alpha, 1 - alpha])
#         if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average
#         self.smooth = smooth  # 1e-6 set '1e-4' when train with FP16
#
#     def forward(self, input, target, weight):
#         # label smooth
#         input = torch.clamp(input, self.smooth, 1.0 - self.smooth)
#
#         if input.dim() > 2:
#             input = input.view(input.size(0), input.size(1), -1)  # N,C,D,H,W => N,C,D*H*W
#             input = input.transpose(1, 2)  # N,C,D*H*W => N,D*H*W,C
#             input = input.contiguous().view(-1, input.size(2))  # N,D*H*W,C => N*D*H*W,C
#         target = target.view(-1, 1)
#
#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1, target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())
#
#         if self.alpha is not None:
#             if self.alpha.type() != input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0, target.data.view(-1))
#             logpt = logpt * Variable(at)
#
#         loss = -1 * (1 - pt) ** self.gamma * logpt
#         if self.size_average:
#             return loss.mean()
#         else:
#             return loss.sum()