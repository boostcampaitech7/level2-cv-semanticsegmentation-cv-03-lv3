import torch
import torch.nn as nn
from transformers import SamModel, SamProcessor

class SAMModel(nn.Module):
    """
    SAM (Segment Anything Model) from Hugging Face
    """
    def __init__(self, model_name: str, **kwargs):
        super(SAMModel, self).__init__()
        self.model = SamModel.from_pretrained(model_name, **kwargs)
        self.processor = SamProcessor.from_pretrained(model_name)

    def forward(self, x: torch.Tensor):
        # SAM requires specific input processing
        inputs = self.processor(images=x, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.pred_masks  # 예측된 마스크 반환

class ModelSelector:
    """
    model을 새롭게 추가하기 위한 방법
        1. model 폴더 내부에 사용하고자하는 custom model 구현
        2. 구현한 Model Class를 model_selector.py 내부로 import
        3. self.model_classes에 아래와 같은 형식으로 추가
        4. yaml파일의 model_name을 설정한 key값으로 변경
    """
    def __init__(self) -> None:
        self.model_classes = {
            "SAM": SAMModel,  # SAM 모델만 남김
        }

    def get_model(self, model_name, **model_parameter):
        return self.model_classes.get(model_name, None)(**model_parameter)
    

# train.py수정
# model_selector = ModelSelector()
# sam_model = model_selector.get_model("SAM", model_name="facebook/sam-vit-huge")