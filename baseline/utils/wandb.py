# wandb.py
import wandb
import os 

def set_wandb(configs, arg):
    wandb.login(key="Your Wandb API")
    wandb.init(
        entity= "naver_cv03",
        project= "Hand Bone Image Segmentation",
        name= os.path.basename(arg),
        config={
                'model': configs['model'],  
                'resize': configs['image_size'],
                'batch_size': configs['train_batch_size'],
                'loss_name': configs['loss'],
                'scheduler_name': configs['scheduler'],
                'learning_rate': configs['learning_rate'],
                'epoch': configs['num_epoch'],
                'random_seed' : configs['random_seed']
            }
    )

