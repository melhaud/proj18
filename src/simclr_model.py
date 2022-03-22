from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

max_epoches = 50

class SimCLRModel(pl.LightningModule):
    def __init__(self, resnet_backbone, img_size : int):
        super().__init__()
        # create a ResNet backbone and remove the classification head
        self.backbone = nn.Sequential(*list(resnet_backbone.children())[:-1])
        #hidden_dim = resnet.fc.in_features
        hidden_dim = resnet_backbone.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, img_size) # last was arg = 128
        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, 50
        )
        
        return [optim], [scheduler]
    