import os
from omegaconf import OmegaConf
import argparse

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import albumentations as A

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import XRayInferenceDataset

import ttach as tta

def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model")
    # config_path를 명령줄 인자로 받을 수 있게 함
    parser.add_argument('--config', type=str, required=True, help="Path to the config.yaml file")
    return parser.parse_args()

def inference_data(TEST_IMAGE_ROOT):
    pngs_inference = {
    os.path.relpath(os.path.join(root, fname), start=TEST_IMAGE_ROOT)
    for root, _dirs, files in os.walk(TEST_IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
    }
    return pngs_inference

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

def main(cfg):
    
    CLASS2IND = {v: i for i, v in enumerate(CLASSES)}
    IND2CLASS = {v: k for k, v in CLASS2IND.items()}

    model = torch.load(os.path.join(cfg.save_dir, cfg.save_file_name))

    def encode_mask_to_rle(mask):
        '''
        mask: numpy array binary mask 
        1 - mask 
        0 - background
        Returns encoded run length 
        '''
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    class ModelOutputWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images):
            outputs = self.model(images)  # 모델 호출
            return outputs[0] # 텐서라면 그대로 반환
            
    def test(model, data_loader, thr=0.5):
        tta_transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1,1.2])
            ]
        )
        model = model.cuda()
        model.eval()
        rles = []
        filename_and_class = []
        with torch.no_grad():

            for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
                images = images.cuda()
                print('TTa start')    
                tta_model = tta.SegmentationTTAWrapper(model,tta_transforms)
                outputs = tta_model(images)
                

                if isinstance(outputs, dict):
                    outputs = outputs['out']
                elif isinstance(outputs, list):
                    outputs = outputs[0]

                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bicubic")
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > thr).detach().cpu().numpy()
                
                for output, image_name in zip(outputs, image_names):
                    for c, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                        
        return rles, filename_and_class
    
    tf = A.Resize(cfg.image_size, cfg.image_size)
    pngs_inference = inference_data(cfg.test_image_root)
    test_dataset = XRayInferenceDataset(transforms=tf,TEST_IMAGE_ROOT=cfg.test_image_root, pngs_inference=pngs_inference)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False
        )
    model = ModelOutputWrapper(model)
    rles, filename_and_class = test(model, test_loader)
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
        })
    
    df.to_csv(cfg.csv_file_name, index=False)


if __name__ == '__main__':

    # 명령줄 인자로 받은 config_path
    args = parse_args()

    # config.yaml 파일 경로
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)

    print(type(cfg.test_image_root))
    print(cfg.test_image_root)
    main(cfg)