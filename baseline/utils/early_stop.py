import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=1, delta=0.5, verbose=False, path='checkpoint.pth', mode='min', epoch = 1):

        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_model_wts = None
        self.best_epoch = epoch

        if mode == 'min':
            self.loss_function = np.less
        elif mode == 'max':
            self.loss_function = np.greater
        else:
            raise ValueError("mode must be either 'min' or 'max'")

    def step(self, score, model, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = model.state_dict()
            print(f'score type: {type(score)},\n' 
                  f'self.delta type: {type(self.delta)},\n'
                  f'self.best_score: {type(self.best_score)}')
            
        elif self.loss_function(score.detach().cpu() + torch.tensor(self.delta, dtype=torch.float32), self.best_score.detach().cpu()):
            self.best_score = score
            self.counter = 0
            self.best_model_wts = model.state_dict()
            self.best_epoch += 1

        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(f'Early stopping triggered. Restoring best model weights from epoch {self.best_epoch}.')
                model.load_state_dict(self.best_model_wts)

                return True
            
        return False
