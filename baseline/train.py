import os
import random
import datetime

import cv2
import numpy as np
from tqdm.auto import tqdm
import albumentations as A
from omegaconf import OmegaConf
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
from utils.wandb import set_wandb

from dataset import XRayDataset
from model import Model_Selector
from loss import Loss_Selector
from scheduler import Scheduler_Selector
from utils.early_stop import EarlyStopping

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model")
    # config_path를 명령줄 인자로 받을 수 있게 함
    parser.add_argument('--config', type=str, required=True, help="Path to the config.yaml file")
    return parser.parse_args()


def train_data(IMAGE_ROOT, LABEL_ROOT):
    pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
    }

    jsons = {
        os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
        for root, _dirs, files in os.walk(LABEL_ROOT)
        for fname in files
        if os.path.splitext(fname)[1].lower() == ".json"
    }

    jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
    pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

    assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
    assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

    pngs = sorted(pngs)
    jsons = sorted(jsons)

    return pngs, jsons


def main(cfg, arg):
    set_wandb(cfg, arg)
    
    if not os.path.exists(cfg.save_dir):                                                           
        os.makedirs(cfg.save_dir)

    tf_pixel = A.OneOf([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.Sharpen(alpha=(0.1, 0.5), lightness=(0.5, 1.5), p=0.5)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
        ], p=1.0)  

    tf_train = A.Compose([
        A.Resize(cfg.image_size, cfg.image_size),     
        tf_pixel         
    ])

    tf_val = A.Resize(cfg.image_size, cfg.image_size)

    pngs, jsons = train_data(cfg.image_root, cfg.label_root)

    train_dataset = XRayDataset(is_train=True, 
                                transforms=tf_train, 
                                IMAGE_ROOT=cfg.image_root, 
                                LABEL_ROOT= cfg.label_root,
                                pngs=pngs, 
                                jsons=jsons
                                )
    
    valid_dataset = XRayDataset(is_train=False, 
                                transforms=tf_val, 
                                IMAGE_ROOT=cfg.image_root,
                                LABEL_ROOT=cfg.label_root,
                                pngs=pngs, 
                                jsons=jsons)

    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=cfg.train_batch_size,
                              shuffle=True,
                              num_workers=cfg.train_batch_size,
                              drop_last=True,
                              )

    valid_loader = DataLoader(dataset=valid_dataset, 
                              batch_size=cfg.val_batch_size,
                              shuffle=False,
                              num_workers=cfg.val_batch_size,
                              drop_last=False
                              )

    # Define function for Training
    def dice_coef(y_true, y_pred):
        y_true = y_true.to('cuda')
        y_pred = y_pred.to('cuda')
        y_true_f = y_true.flatten(2)
        y_pred_f = y_pred.flatten(2)
        intersection = torch.sum(y_true_f * y_pred_f, -1)
        
        eps = 0.0001
        return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

    def save_model(model, file_name=cfg.save_file_name):
        output_path = os.path.join(cfg.save_dir, file_name)
        torch.save(model, output_path)

    def set_seed():
        torch.manual_seed(cfg.random_seed)
        torch.cuda.manual_seed(cfg.random_seed)
        torch.cuda.manual_seed_all(cfg.random_seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(cfg.random_seed)
        random.seed(cfg.random_seed)

    # validation
    def validation(epoch, model, data_loader, criterion, thr=0.5):
        print(f'Start validation #{epoch:2d}')
        model.eval()

        dices = []
        with torch.no_grad():
            total_loss = 0
            cnt = 0

            for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
                images, masks = images.cuda(), masks.cuda()         
                model = model.cuda()
                
                outputs = model(images)

                if isinstance(outputs, dict):
                    outputs = outputs['out']
                elif isinstance(outputs, list):
                    outputs = outputs[0]
            
                output_h, output_w = outputs.size(-2), outputs.size(-1)
                mask_h, mask_w = masks.size(-2), masks.size(-1)
                
                # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
                if output_h != mask_h or output_w != mask_w:
                    outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
                
                loss = criterion(outputs, masks)
                total_loss += loss
                cnt += 1
                
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > thr).detach().cpu()
                masks = masks.detach().cpu()
                
                dice = dice_coef(outputs, masks)
                dices.append(dice.detach().cpu())
                    
        dices = torch.cat(dices, 0)
        dices_per_class = torch.mean(dices, 0)
        dice_str = [
            f"{c:<12}: {d.item():.4f}"
            for c, d in zip(CLASSES, dices_per_class)
        ]
        dice_str = "\n".join(dice_str)
        print(dice_str)
        
        avg_dice = torch.mean(dices_per_class).item()
        
        return avg_dice, dices_per_class, total_loss / len(data_loader)

    # Train
    def train(model, data_loader, val_loader, criterion, optimizer, scheduler, accumulation_steps=cfg.accumulation_steps, patience=cfg.patience):
        print(f'Start training..')

        best_dice = 0.
        scaler = GradScaler()
        cos_scheduler = scheduler

        # EarlyStopping 객체 초기화
        early_stopping = EarlyStopping(patience=patience, mode='min', verbose=True, path=cfg.save_file_name)

        for epoch in range(cfg.num_epoch):
            model.train()

            for step, (images, masks) in enumerate(data_loader):            
                images, masks = images.cuda(), masks.cuda()
                model = model.cuda()

                optimizer.zero_grad()

                with autocast():
                    outputs = model(images)

                    if isinstance(outputs, dict):
                        outputs = outputs['out']
                    elif isinstance(outputs, list):
                        outputs = outputs[0]

                    loss = criterion(outputs, masks)

                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(data_loader):
                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad()

                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    lr = optimizer.param_groups[0]['lr']
                else:
                    lr = scheduler.get_last_lr()[0]

                wandb.log({
                    "Epoch" : epoch,
                    "train/loss": loss.item(),
                    "train/learning_rate": lr
                }, step=epoch)

                if (step + 1) % 25 == 0:
                    print(
                        f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                        f'Epoch [{epoch+1}/{cfg.num_epoch}], '
                        f'Step [{step+1}/{len(data_loader)}], '
                        f'Loss: {round(loss.item(),4)}'
                    )

            if epoch < 0:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True, factor=0.5)
                scheduler.step(loss)
            else:
                scheduler = cos_scheduler
                scheduler.step()

            if (epoch + 1) % cfg.val_every == 0:
                dice,  dices_per_class, val_loss = validation(epoch + 1, model, val_loader, criterion)

                wandb.log({
                    "validation Loss": val_loss,
                    "Avg dice_score": dice,
                    **{f"Dice/{class_name}": dice.item() for class_name, dice in zip(CLASSES, dices_per_class)},
                }, step=epoch)

                if best_dice < dice:
                    print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                    print(f"Save model in {cfg.save_dir}")
                    best_dice = dice
                    save_model(model)

                    wandb.save(f"{cfg.save_dir}/best_model.pth")

            # # EarlyStopping 체크
            # if early_stopping.step(loss, model, epoch):
            #     print(f'Early stopping at epoch {epoch+1}')
            #     break  # 학습 종료


    # Setting
    print(f"Selected model: {cfg.model}")
    print(f"Selected loss function: {cfg.loss}")
    print(f"Selected scheduler: {cfg.scheduler}")
    
    model = Model_Selector().get_model(cfg.model)
    criterion = Loss_Selector().get_loss(cfg.loss)
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.learning_rate, weight_decay=1e-6)
    scheduler = Scheduler_Selector(num_epoch=cfg.num_epoch).get_scheduler(cfg.scheduler, optimizer=optimizer)
    
    # model_tk = ModelSelector()
    # model_tk.get_model('SAM')
    
    set_seed()

    # Run
    train(model, train_loader, valid_loader, criterion, optimizer, scheduler)

if __name__ == '__main__':

    # 명령줄 인자로 받은 config_path
    args = parse_args()

    # config.yaml 파일 경로
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)

    main(cfg, args.config)