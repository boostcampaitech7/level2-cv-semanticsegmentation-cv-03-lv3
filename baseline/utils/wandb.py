# wandb.py
import wandb

def set_wandb(configs):
    wandb.login(key="5dbe719a1633f39d630190b12a8c1fc3f311faa6")
    wandb.init(
        entity= "naver_cv03",
        project= "Hand Bone Image Segmentation",
        name= "test",
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


# train_batch_size :  X 사용 안함         
# val_batch_size :    X 사용 안함
# learning_rate :    사용
# random_seed :      사용

# model :            사용
# num_epoch :        사용 -> 
# val_every :        X 사용 안함
# loss :             사용 -> loss_name
# scheduler :        사용 -> scheduler_name
    
# save_dir :         X 사용 안함 
# save_file_name :   X 사용 안함
# csv_file_name :    X 사용 안함

# project_name : '실험 주제'         X 사용 안함
# detail : '세부내용'                X 사용 안함



