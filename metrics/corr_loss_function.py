import torch
import torch.nn as nn
import torch.nn.functional as F
# cofrom fast_soft_sort.pytorch_ops import soft_rank


class CorrLoss(nn.Module):
    def __init__(self):
        super(CorrLoss, self).__init__()

    def forward(self, input, target):
        input_n = input - input.mean()
        target_n = target - target.mean()
        input_n = input_n / input_n.norm()
        target_n = target_n / target_n.norm()
        return (input_n * target_n).sum()


class SpearmanLoss(nn.Module):
    def __init__(self):
        super(SpearmanLoss, self).__init__()

    def forward(self, input_, target, regularization='l2', regularization_strength=1.0):
        try:
            if input_.get_device() == 0:
                input_ = input_.cpu()
        except RuntimeError:
            pass
        input_ = soft_rank(
            input_,
            regularization=regularization,
            regularization_strength=regularization_strength
        )
        try:
            if input_.get_device() == -1:
                input_ = input_.cuda()
        except RuntimeError:
            input_ = input_.cuda()
        return CorrLoss()(input_, target)
