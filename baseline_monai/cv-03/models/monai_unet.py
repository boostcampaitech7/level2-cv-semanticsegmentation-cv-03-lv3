import torch
import torch.nn as nn
from monai.networks.nets import UNet as MonaiUNet  # 이름 충돌을 피하기 위해 MonaiUNet 사용

class Monai_UNet(nn.Module):
    """
    Base Model UNet using MONAI
    """
    def __init__(self,
                 **kwargs):
        super(Monai_UNet, self).__init__()
        self.model = MonaiUNet(**kwargs)

    def forward(self, x: torch.Tensor):
        return self.model(x)