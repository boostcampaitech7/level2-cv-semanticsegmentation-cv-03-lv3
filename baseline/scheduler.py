import torch.optim as optim
from omegaconf import OmegaConf

class Scheduler_Selector():
    def __init__(self, num_epoch):
        # 각 스케줄러 함수들을 클래스 메소드로 포함
        self.classes = {
            'MultiStepLR': self.multi_step_lr,
            'CosineAnnealingLR': self.cosine_annealing_lr,
            'CosineAnnealingWarmRestarts': self.cosine_annealing_warm_restarts
        }
        self.num_epoch = num_epoch

    def multi_step_lr(self, optimizer, **scheduler_params):
        """ MultiStepLR 스케줄러 생성 """
        return optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 80], gamma=0.1)

    def cosine_annealing_lr(self, optimizer, num_epoch = None, **scheduler_params):
        """ CosineAnnealingLR 스케줄러 생성 """
        num_epoch = num_epoch or self.num_epoch
        return optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max= num_epoch, eta_min= 1e-6, last_epoch= -1)

    def cosine_annealing_warm_restarts(self, optimizer, **scheduler_params):
        """ CosineAnnealingWarmRestarts 스케줄러 생성 """
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=2, eta_min=1e-6, last_epoch=-1)

    def get_scheduler(self, scheduler_name, optimizer, **scheduler_params):
        """ 주어진 이름에 맞는 스케줄러 반환 """
        # 클래스에서 정의된 스케줄러 이름을 검색하여 호출
        scheduler_fn = self.classes.get(scheduler_name, None)

        if scheduler_fn is None:
            raise ValueError(f"Scheduler '{scheduler_name}' not found. Available schedulers: {list(self.classes.keys())}")
        
        return scheduler_fn(optimizer, **scheduler_params)
