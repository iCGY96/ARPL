import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.Dist import Dist

class GCPLoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(GCPLoss, self).__init__()
        self.weight_pl = options['weight_pl']
        self.temp = options['temp']
        self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['feat_dim']) # 

    def forward(self, x, y, labels=None):
        dist = self.Dist(x)
        logits = F.softmax(-dist, dim=1)
        if labels is None: return logits, 0
        loss = F.cross_entropy(-dist / self.temp, labels)
        center_batch = self.Dist.centers[labels, :]
        loss_r = F.mse_loss(x, center_batch) / 2
        loss = loss + self.weight_pl * loss_r

        return logits, loss