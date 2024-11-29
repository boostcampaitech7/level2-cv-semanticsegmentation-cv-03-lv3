import torch
import torch.nn as nn
from monai.networks.nets import BasicUNetPlusPlus
from torchvision import models
import segmentation_models_pytorch as smp
from model_tk import SamVitHugeModel


class Model_Selector:
    def __init__(self):
        self.model_classes = {
            'unetplusplus': self.unet_plus_plus,
            'deeplabv3_resnet101': self.deeplab_v3_model,
            'unetplusplus_smp': self.unetplusplus_smp,
            'SAM': SamVitHugeModel
        }

    def unet_plus_plus(self, in_channels=3, out_channels=29, features=(64, 128, 256, 512, 1024, 1024)):
        """ monai의 BasicUNetPlusPlus 모델 생성 """
        model = BasicUNetPlusPlus(
            spatial_dims=2,               # 이미지 차원 (2D 또는 3D)
            in_channels=in_channels,      # 입력 채널 수
            out_channels=out_channels,    # 출력 채널 수
            features=features             # 각 레벨의 필터 수
        )
        return model

    def deeplab_v3_model(self, pretrained=True, num_classes=29):
        """ torchvision의 DeepLabV3 모델 생성 """
        model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        return model
    
    def unetplusplus_smp(self, encoder_name="timm-efficientnet-b0", num_classes=29):
        """ segmentation_models.pytorch의 Unet++ 모델 생성 """
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,  # Encoder 모델 선택 (예: resnet34, resnet50)
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes
        )
        return model

    def get_model(self, model_name, **model_params):
        """ 주어진 model_name에 맞는 모델을 반환 """
        model_fn = self.model_classes.get(model_name, None)

        if model_fn is None:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.model_classes.keys())}")
        
        return model_fn(**model_params)
