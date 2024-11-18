import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, path='checkpoint.pth', mode='max'):
        """
        :param patience: 성능 향상이 없을 때 기다릴 에포크 수
        :param delta: 성능 향상 최소 기준
        :param verbose: 출력 여부
        :param path: 모델 체크포인트 경로
        :param mode: 'min'일 경우, 최소화하려는 값, 'max'일 경우 최대화하려는 값
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_model_wts = None

        if mode == 'min':
            self.loss_function = np.less
        elif mode == 'max':
            self.loss_function = np.greater
        else:
            raise ValueError("mode must be either 'min' or 'max'")

    def step(self, score, model):
        """
        Early stopping을 체크하고, 성능 향상이 없으면 중단하도록 함.
        
        :param score: 현재 성능 (e.g., validation loss or metric)
        :param model: 현재 모델
        :return: True if early stopping triggered, otherwise False
        """
        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = model.state_dict()
        elif self.loss_function(score + self.delta, self.best_score):
            self.best_score = score
            self.counter = 0
            self.best_model_wts = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(f'Early stopping triggered. Restoring best model weights from epoch {self.best_epoch}.')
                model.load_state_dict(self.best_model_wts)
                return True
        return False
