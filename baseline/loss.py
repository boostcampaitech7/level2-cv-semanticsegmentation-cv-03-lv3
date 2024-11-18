import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss_Selector():
    def __init__(self):
        self.loss_classes = {
            'BCEWithLogitLoss': self.bcewithlogitloss,
            'focal_loss': self.focal_loss,
            'dice_loss': self.dice_loss,
            'iou_loss': self.iou_loss,
            'bcedice_loss': self.bcedice_loss,
            'diceiou_loss': self.diceiou_loss
        }

    def bcewithlogitloss(self):
        """ BCEWithLogitLoss 함수 """
        return nn.BCEWithLogitsLoss()

    def focal_loss(self, alpha=0.25, gamma=2):
        """ Focal Loss 함수 """
        def focalloss(inputs, targets):
            inputs = torch.sigmoid(inputs)
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
            BCE_EXP = torch.exp(-BCE)
            loss = alpha * (1 - BCE_EXP) ** gamma * BCE
            return loss
        return focalloss

    def dice_loss(self, smooth=1.):
        """ Dice Loss 함수 """
        def diceloss(pred, target):
            pred = pred.contiguous()
            target = target.contiguous()
            intersection = (pred * target).sum(dim=2).sum(dim=2)
            loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
            return loss.mean()
        return diceloss

    def iou_loss(self, smooth=1):
        """ IOU Loss 함수 """
        def iouloss(inputs, targets):
            inputs = torch.sigmoid(inputs)
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            intersection = (inputs * targets).sum()
            total = (inputs + targets).sum()
            union = total - intersection
            IoU = (intersection + smooth) / (union + smooth)
            return 1 - IoU
        return iouloss

    def bcedice_loss(self, bce_weight=0.5):
        """ BCEDice Loss 함수 """
        def bcediceloss(pred, target):
            bce = F.binary_cross_entropy_with_logits(pred, target)
            pred = torch.sigmoid(pred)
            dice = self.dice_loss()(pred, target)
            loss = bce * bce_weight + dice * (1 - bce_weight)
            return loss
        return bcediceloss

    def diceiou_loss(self, iou_weight=0.5):
        """ DiceIOU Loss 함수 """
        def diceiouloss(pred, target):
            iouloss = self.iou_loss()
            iou = iouloss(pred, target)
            pred = torch.sigmoid(pred)
            dice = self.dice_loss()(pred, target)
            loss = iou * iou_weight + dice * (1 - iou_weight)
            return loss
        return diceiouloss

    def get_loss(self, loss_name, **loss_params):
        """ 선택한 손실 함수 반환 """
        loss_fn = self.loss_classes.get(loss_name, None)

        if loss_fn is None:
            raise ValueError(f"Loss '{loss_name}' not found. Available losses: {list(self.loss_classes.keys())}")
        
        return loss_fn(**loss_params)

