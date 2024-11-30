import torch
import torch.nn as nn
from monai.networks.nets import BasicUNetPlusPlus as MonaiUNetplusplus  # 이름 충돌을 피하기 위해 MonaiUNetplusplus 사용

class Monai_UNetplusplus(nn.Module):
    """
    Base Model UNet_plusplus using MONAI
    """
    def __init__(self,
                 **kwargs):
        super(Monai_UNetplusplus, self).__init__()

        # features 파라미터를 int 튜플로 변환
        if 'features' in kwargs:
            if isinstance(kwargs['features'], (list, tuple)):
                kwargs['features'] = tuple(int(x) for x in kwargs['features'])
            elif isinstance(kwargs['features'], str):
                # 문자열에서 숫자만 추출하여 튜플로 변환
                import re
                numbers = re.findall(r'\d+', kwargs['features'])
                kwargs['features'] = tuple(int(x) for x in numbers)

        self.model = MonaiUNetplusplus(**kwargs)

    def forward(self, x: torch.Tensor):
        return self.model(x)