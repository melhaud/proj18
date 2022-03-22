from lightly.models.modules import BarlowTwinsProjectionHead
from my_resnet import resnet20
from torch import nn
from utils import custom_collate_fn, get_classes


class BarlowTwins(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(64, 2048, 512) # orig: (64,2048,2048)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        # embedding from backbone ResNet20
        self.backbone_embedding = x
        z = self.projection_head(x)
        # embedding from projection head
        self.proj_embedding = z
        return z
