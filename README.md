# Hand Bone Image Segmentation

## 🥇 팀 구성원

<div align="center">
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/kupulau">
        <img src="https://aistages-api-public-prod.s3.amazonaws.com/app/Users/00003808/user_image.png" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>황지은</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/asotea">
        <img src="https://aistages-api-public-prod.s3.amazonaws.com/app/Users/00003808/user_image.png" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>김태균</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/mujjinungae">
        <img src="https://aistages-api-public-prod.s3.amazonaws.com/app/Users/00003808/user_image.png" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>이진우</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/glasshong">
        <img src="https://aistages-api-public-prod.s3.amazonaws.com/app/Users/00003808/user_image.png" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>홍유리</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/EuiInSeong">
        <img src="https://aistages-api-public-prod.s3.amazonaws.com/app/Users/00003808/user_image.png" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>성의인</b></sub><br />
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/jinbong-yeom">
        <img src="https://aistages-api-public-prod.s3.amazonaws.com/app/Users/00003808/user_image.png" width="120px" height="120px" alt=""/>
        <hr />
        <sub><b>염진봉</b></sub><br />
      </a>
    </td>
  </tr>
</table>
</div>

<br />

## 🗒️ 프로젝트 개요

뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적입니다.

Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다.

1. 질병 진단의 목적으로 뼈의 형태나 위치가 변형되거나 부러지거나 골절 등이 있을 경우, 그 부위에서 발생하는 문제를 정확하게 파악하여 적절한 치료를 시행할 수 있습니다.

2. 수술 계획을 세우는데 도움이 됩니다. 의사들은 뼈 구조를 분석하여 어떤 종류의 수술이 필요한지, 어떤 종류의 재료가 사용될 수 있는지 등을 결정할 수 있습니다.

3. 의료장비 제작에 필요한 정보를 제공합니다. 예를 들어, 인공 관절이나 치아 임플란트를 제작할 때 뼈 구조를 분석하여 적절한 크기와 모양을 결정할 수 있습니다.

4. 의료 교육에서도 활용될 수 있습니다. 의사들은 병태 및 부상에 대한 이해를 높이고 수술 계획을 개발하는 데 필요한 기술을 연습할 수 있습니다.

여러분에 의해 만들어진 우수한 성능의 모델은 질병 진단, 수술 계획, 의료 장비 제작, 의료 교육 등에 사용될 수 있을 것으로 기대됩니다. 🌎

<br />

## 📅 프로젝트 일정

프로젝트 전체 일정

- 2024.11.11 (월) 10:00 ~ 2024.11.28 (목) 19:00

프로젝트 세부 일정

![schedule](https://github.com/user-attachments/assets/fad69118-9c1d-4c84-884f-d74df9e8c543)

## 💻 개발 환경

```bash
- Language : Python
- Environment
  - CPU : Intel(R) Xeon(R) Gold 5120
  - GPU : Tesla V100-SXM2 32GB × 1
- Framework : PyTorch
- Collaborative Tool : Git, Wandb, Notion
```

## 📁 데이터셋 구조

```
📦data
  ├─📂test
  │    └─📂DCM
  │         ├─📂ID040
  │         │     📜image1661319116107.png
  │         │     📜image1661319145363.png
  │         └─📂ID041
  │               📜image1661319356239.png
  │               📜image1661319390106.png
  │
  ├─📂train
  │    ├─📂DCM
  │    │   ├─📂ID001
  │    │   │     📜image1661130828152_R.png
  │    │   │     📜image1661130891365_L.png
  │    │   └─📂ID002
  │    │         📜image1661144206667.png
  │    │         📜image1661144246917.png
  │    │        
  │    └─📂outputs_json
  │               ├─📂ID001
  │               │     📜image1661130828152_R.json
  │               │     📜image1661130891365_L.json
  │               └─📂ID002
                        📜image1661144206667.json
                        📜image1661144246917.json
```

- 학습에 사용할 이미지는 800개, 추론에 사용할 이미지는 288개로 각각 data/train/, data/test 아래에 저장되어 있습니다.
- 제공되는 이미지 데이터셋은 손가락, 손등, 팔 부위의 29개 종류의 뼈가 찍힌 2048 x 2048 크기의 사진으로 구성되어 있습니다.
- json 파일은 각 학습 이미지의 뼈 종류를 구분한 annotation file입니다.

<br />

## 📁 프로젝트 구조 

```
📦level2-cv-semanticsegmentation-cv-03-lv3
 ┣ 📂.github
 ┃ ┗ 📂ISSUE_TEMPLATE
 ┃   ┗ 📜bug_report_template.yaml
 ┃   ┗ 📜documentation_issue_template.yaml
 ┃   ┗ 📜enhancement_request_template.yaml
 ┃   ┗ 📜feature_request_template.yaml
 ┃ ┗ 📜.keep
 ┃ ┗ 📜pull_request_templat.md
 ┣ 📂EDA
 ┃ ┗ 📜EDA.ipynb
 ┣ 📂baseline
 ┃ ┗ 📂config
 ┃   ┗ 📜base_config.yaml
 ┃   ┗ 📜setting.txt
 ┃ ┗ 📂utils
 ┃   ┗ 📂ensemble_input
 ┃     ┗ 📜9542.csv
 ┃     ┗ 📜9680.csv
 ┃   ┗ 📂wandb
 ┃     ┗ 📜wandb.ipynb
 ┃   ┗ 📜classwise_ensemble.py
 ┃   ┗ 📜crop_to_2048.ipynb
 ┃   ┗ 📜early_stop.py
 ┃   ┗ 📜hard_ensemble.py
 ┃   ┗ 📜visualize_test.ipynb
 ┃   ┗ 📜visualize_train.ipynb
 ┃   ┗ 📜wandb.py
 ┃ ┗ 📜dataset.py
 ┃ ┗ 📜inference.py
 ┃ ┗ 📜loss.py
 ┃ ┗ 📜model.py
 ┃ ┗ 📜model_tk.py
 ┃ ┗ 📜model_ui.py
 ┃ ┗ 📜scheduler.py
 ┃ ┗ 📜train.py
 ┣ 📂baseline_monai/cv-03
 ┃ ┗ 📂configs
 ┃   ┗ 📜base_train.yaml
 ┃ ┗ 📂loss
 ┃   ┗ 📜base_loss.py
 ┃   ┗ 📜loss_selector.py
 ┃ ┗ 📂models
 ┃   ┗ 📜base_model.py
 ┃   ┗ 📜model_selector.py
 ┃   ┗ 📜monai_unet.py
 ┃   ┗ 📜monai_unetplusplus.py
 ┃ ┗ 📂scheduler
 ┃   ┗ 📜scheduler_selector.py
 ┃ ┗ 📂utils
 ┃   ┗ 📜wandb.py
 ┃ ┗ 📜dataset.py
 ┃ ┗ 📜inference.py
 ┃ ┗ 📜train.py
 ┃ ┗ 📜trainer.py
 ┣ 📜.gitignore
 ┣ 📜Capitate_crop.ipynb
 ┣ 📜README.md
 ┣ 📜SAM_sample.ipynb
```

### baseline code 설명

#### 1) `train.py`

- argparse를 통해 설정 값을 받아 모델 학습을 수행하는 스크립트

#### 2) `model.py`

- 모델을 정의한 파일
- Model_Selector 클래스로 UNet++, DeepLabV3 등의 모델 선택 가능

#### 3) `inference.py`

- 학습된 모델을 불러와 예측값을 반환하는 스크립트

#### 4) `dataset.py`

- 학습 및 추론 데이터를 로드하는 클래스를 정의한 파일

#### 5) `loss.py`

- 학습 시에 사용하는 다양한 loss를 정의하고 선택하는 스크립트

#### 6) `scheduler.py`

- 학습 시에 사용하는 다양한 scheduler를 정의하고 선택하는 스크립트

<br />

## ⚙️ requirements

- opencv-python-headless==4.10.0.84
- pandas==2.2.3
- scikit-learn==1.5.2
- albumentations==1.4.18
- matplotlib==3.9.2
- os
- random
- datetime
- cv2
- numpy
- tqdm
- omegaconf
- argparse
- torch
- wandb

`pip install -r requirements.txt`

<br />

## ▶️ 실행 방법

#### 학습 및 체크포인트 저장

`python train.py --config base_config.yaml`

#### 추론

`python inference.py --config base_config.yaml`

#### `base_config.yaml` 설명

<details>
<summary>클릭해서 펼치기/접기</summary>

1. **`Data root`**:
   - **설명**: 학습 이미지, annotation 데이터가 저장된 디렉토리 경로를 설정합니다.
   - **예시**: image_root: `/data/ephemeral/home/level2-cv-semanticsegmentation-cv-03-lv3/data/train/DCM`
              label_root: `/data/ephemeral/home/level2-cv-semanticsegmentation-cv-03-lv3/data/train/outputs_json`

2. **`Hyperparameter`**:

   - **설명**: 학습 시 사용되는 하이퍼 파라미터(배치 사이즈, 학습률, random seed)를 설정합니다.
   - **예시**: train_batch_size: 4
              val_batch_size: 2
              learning_rate: 1e-4
              random_seed: 42

3. **`Train`**:

   - **설명**: 학습 시 사용하는 모델, epoch, validation 주기, 손실 함수, 스케줄러, 이미지 사이즈, accumulation step, patience를 설정합니다.
   - **예시**: model: 'unetplusplus'
              num_epoch: 60
              val_every: 5
              loss: 'diceiou_loss'
              scheduler: 'CosineAnnealingWarmRestarts'
              image_size: 1024
              accumulation_steps: 4
              patience: 3

4. **`Test`**:

   - **설명**: 추론 이미지가 저장된 디렉토리 경로를 설정합니다.
   - **예시**: test_image_root: "/data/ephemeral/home/level2-cv-semanticsegmentation-cv-03-lv3/data/test/DCM"

5. **`Save directory`**:

   - **설명**: 학습된 모델, 추론 결과를 저장할 경로와 파일 이름을 설정합니다.
   - **예시**: save_dir: "checkpoints"
              save_file_name: "unet++_diceiou_best_model.pt"
              csv_file_name: "output_unet++_diceiou.csv"

6. **`Project name`**:

   - **설명**: 실험 주제와 설명을 기술합니다.
   - **예시**: project_name: '실험 주제'
              detail: '세부 내용'

</details>

<br />