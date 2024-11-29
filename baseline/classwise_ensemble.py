import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

# RLE로 인코딩된 결과를 mask map으로 복원합니다. (RLE -> mask map)
def decode_rle_to_mask(rle, height, width):
    s = str(rle).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)

# CSV 파일들이 저장된 폴더 경로
# 앙상블할 폴더들을 ensemble_input에 넣어주면 됩니다.
input_path = "./ens_in"
path=glob.glob(f"{input_path}/*.csv")
dfs=[pd.read_csv(file) for file in path]
print(dfs[0])
print(dfs[1])
# 29 * 288
# df length를 29개씩 끊어서 처리
for i,k in zip(tqdm(range(0, len(dfs[0]), 29)),tqdm(range(0, len(dfs[1]), 8))):#한 이미지씩 
    # class number
    for j in range(29): # 한 이미지의 한 클래스씩

        result = np.zeros((2048, 2048), dtype=int)# 이미지 빈칸

        # ensemble candidates
        for idx, df in enumerate(dfs): #결과 하나씩 불러옴
            if idx==0 and j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 27, 28]:
                rle = df.iloc[i+j].rle #i는 시작점 j는 클래스 0+0 =finger-1 .rle = 한 행
                mask = decode_rle_to_mask(rle, 2048, 2048) # 01 matrix 마스크픽셀 식으로 바꾸기
                result += mask #복원 2모델이 같은 예측을 했다면 2
                break
            elif idx==1 and j in [19, 20, 21, 22, 23, 24, 25, 26]:
                if j==19: rle = df.iloc[k+0].rle #i는 시작점 j는 클래스 0+0 =finger-1 .rle = 한 행
                elif j==20: rle = df.iloc[k+1].rle
                elif j==21: rle = df.iloc[k+2].rle
                elif j==22: rle = df.iloc[k+3].rle
                elif j==23: rle = df.iloc[k+4].rle
                elif j==24: rle = df.iloc[k+5].rle
                elif j==25: rle = df.iloc[k+6].rle
                elif j==26: rle = df.iloc[k+7].rle
                
                mask = decode_rle_to_mask(rle, 2048, 2048) # 01 matrix 마스크픽셀 식으로 바꾸기
                result += mask #복원 2모델이 같은 예측을 했다면 2
                break
                
            
        # calculate with threshold
        dfs[0].rle[i+j] = encode_mask_to_rle(result) # 다시 rle로 바꿈
        
    
        
dfs[0].to_csv("mixoutput.csv", index=False)
