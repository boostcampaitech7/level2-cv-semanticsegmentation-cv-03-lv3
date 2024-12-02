import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForMaskGeneration


class ModelSelector():
    def __init__(self) -> None:
        self.model_classes = {
            "Sam": SamVitHugeModel,
        }

    def get_model(self, model_name):
        model_class = self.model_classes.get(model_name, None)
        return self.model_classes.get(model_name, None)



class SamVitHugeModel(nn.Module):

    def __init__(self, num_classes : int = 29):
        super(SamVitHugeModel, self).__init__()
        self.processor = AutoProcessor.from_pretrained("facebook/sam-vit-huge")
        self.model = AutoModelForMaskGeneration.from_pretrained("facebook/sam-vit-huge")


        if hasattr(self.model, 'mask_decoder') and hasattr(self.model.mask_decoder, 'upscale_conv2'):
            in_channels = self.model.mask_decoder.upscale_conv2.in_channels
            self.model.mask_decoder.upscale_conv2 = nn.ConvTranspose2d(in_channels, num_classes, kernel_size=(2, 2), stride=(2, 2))
        else:
            raise AttributeError("The model's mask_decoder or upscale_conv2 layer not found.")
    
    def forward(self, images :torch.Tensor, masks = torch.Tensor):
        inputs = self.processor(images, return_tensors = 'pt')

        return self.model(**inputs, labels = masks)
    





    