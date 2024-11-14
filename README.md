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

![스크린샷 2024-11-13 오후 11 48 45](https://github.com/user-attachments/assets/44c4e493-47e0-4448-98b7-d758e6d1afb1)

## 💻 개발 환경

```bash
- Language : Python
- Environment
  - CPU : Intel(R) Xeon(R) Gold 5120
  - GPU : Tesla V100-SXM2 32GB × 1
- Framework : PyTorch
- Collaborative Tool : Git, Wandb, Notion
```

## 🏆 프로젝트 결과

- Public
- Private 

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
```

#### 1) `train.py`

- 모델 학습을 수행하는 함수

#### 2) `seed.py`

- 모든 랜덤 연산에서 동일한 결과를 재현할 수 있도록 시드를 설정하는 파일
- random, numpy, torch 라이브러리와 관련된 시드 설정 및 CUDA 관련 고정 설정

#### 3) `model.py`

- 모델을 정의한 파일
- 입력된 데이터를 모델에 전달하여 예측을 수행하는 forward 메서드 포함

#### 4) `main.py`

- 학습과 추론을 위한 메인 스크립트로, argparse를 통해 설정 값을 받아 모델 학습과 추론을 수행

#### 5) `inference.py`

- 예측값을 반환

#### 6) `dataset.py`

- 학습 및 추론 데이터를 로드하는 CustomDataset 클래스를 정의한 파일
- 이미지 데이터를 로드하고, 주어진 변환(transform)을 적용하여 반환하며, 학습 또는 추론 모드에 따라 라벨과 함께 데이터를 반환

#### 7) `augmentation.py`

- 다양한 데이터 증강 기법을 적용하는 augmentation 클래스 정의

<br />

## ⚙️ requirements

- opencv-python-headless==4.10.0.84
- pandas==2.2.3
- scikit-learn==1.5.2
- albumentations==1.4.18
- matplotlib==3.9.2

`pip install -r requirements.txt`

<br />

## ▶️ 실행 방법

#### 학습 및 체크포인트 저장

`python main.py --train_dir ../data/train --train_csv ../data/train.csv --test_dir ../data/test --test_csv ../data/test.csv --batch_size 16 --resize_height 448 --resize_width 448 --learning_rate 1e-4 --max_epochs 50`

#### 체크포인트에서 학습 재개

`python main.py --train_dir ../data/train --train_csv ../data/train.csv --test_dir ../data/test --test_csv ../data/test.csv --resume_training --batch_size 16 --resize_height 448 --resize_width 448`

#### `argparse` 인자 설명

<details>
<summary>클릭해서 펼치기/접기</summary>

1. **`--train_dir` (필수 인자)**:
   - **설명**: 학습 데이터가 저장된 디렉토리 경로를 설정합니다.
   - **예시**: `--train_dir ../data/train`

2. **`--train_csv` (필수 인자)**:

   - **설명**: 학습 데이터의 이미지 경로와 레이블이 포함된 CSV 파일 경로를 설정합니다.
   - **예시**: `--train_csv ../data/train.csv`

3. **`--test_dir` (필수 인자)**:

   - **설명**: 테스트 데이터가 저장된 디렉토리 경로를 설정합니다.
   - **예시**: `--test_dir ../data/test`

4. **`--test_csv` (필수 인자)**:

   - **설명**: 테스트 데이터의 이미지 경로와 ID가 포함된 CSV 파일 경로를 설정합니다.
   - **예시**: `--test_csv ../data/test.csv`

5. **`--save_dir` (선택적 인자, 기본값: `./model_checkpoints`)**:

   - **설명**: 학습된 모델 체크포인트를 저장할 디렉토리 경로를 설정합니다.
   - **예시**: `--save_dir ./checkpoints`

6. **`--log_dir` (선택적 인자, 기본값: `./training_logs`)**:

   - **설명**: 학습 로그를 저장할 디렉토리 경로를 설정합니다.
   - **예시**: `--log_dir ./logs`

7. **`--batch_size` (선택적 인자, 기본값: `32`)**:

   - **설명**: 학습과 추론 시 사용할 배치 크기를 설정합니다.
   - **예시**: `--batch_size 16`

8. **`--learning_rate` (선택적 인자, 기본값: `1e-5`)**:

   - **설명**: 학습 시 사용하는 학습률을 설정합니다.
   - **예시**: `--learning_rate 0.001`

9. **`--weight_decay` (선택적 인자, 기본값: `0.01`)**:

   - **설명**: 옵티마이저에서 사용하는 가중치 감소값을 설정합니다.
   - **예시**: `--weight_decay 0.001`

10. **`--max_epochs` (선택적 인자, 기본값: `50`)**:

    - **설명**: 학습할 최대 epoch 수를 설정합니다.
    - **예시**: `--max_epochs 100`

11. **`--accumulation_steps` (선택적 인자, 기본값: `8`)**:

    - **설명**: 그래디언트 누적을 위한 스텝 수를 설정합니다.
    - **예시**: `--accumulation_steps 4`

12. **`--patience` (선택적 인자, 기본값: `5`)**:

    - **설명**: 학습 중 조기 종료(Early Stopping)를 위한 patience를 설정합니다. 이 값은 검증 손실이 개선되지 않을 때 몇 번의 에포크를 더 실행할지 결정합니다.
    - **예시**: `--patience 10`

13. **`--resume_training` (선택적 인자)**:

    - **설명**: 가장 최근의 체크포인트에서 학습을 재개할지 여부를 설정합니다. 이 플래그를 추가하면, 학습이 중단된 체크포인트에서 이어서 학습이 가능합니다.
    - **예시**: `--resume_training`

14. **`--resize_height` (선택적 인자, 기본값: `448`)**:

    - **설명**: 이미지 변환 시 이미지의 높이를 설정합니다.
    - **예시**: `--resize_height 512`

15. **`--resize_width` (선택적 인자, 기본값: `448`)**:
    - **설명**: 이미지 변환 시 이미지의 너비를 설정합니다.
    - **예시**: `--resize_width 512`

</details>

<br />

## ✏️ Wrap-Up Report
