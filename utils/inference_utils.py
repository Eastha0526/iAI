import pickle
import numpy as np
import xgboost as xgb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
# ---
from utils import utils

# Encoder 및 Scaler 불러오기
def load_preprocessors(encoder_path, scaler_path):
    """
    저장된 encoder와 scaler를 불러옵니다.
    
    Args:
        encoder_path (str): encoder.pkl 파일 경로
        scaler_path (str): scaler.pkl 파일 경로
    
    Returns:
        encoder, scaler: 불러온 LabelEncoder 및 StandardScaler 객체
    """
    with open(encoder_path, 'rb') as enc_file:
        try:
            encoder = pickle.load(enc_file)
        except FileNotFoundError as e:
            print(f"Error: File not found. Details: {e}")
            print(f"Encoder 데이터 로드 실패")
            raise
    
    with open(scaler_path, 'rb') as scl_file:
        try:
            scaler = pickle.load(scl_file)
        except FileNotFoundError as e:
            print(f"Error: File not found. Details: {e}")
            print(f"Scaler 데이터 로드 실패")
            raise
    
    return encoder, scaler

# 추론 모델에 필요한 데이터 구성
def make_inference_model_data(info, test_x_s):
    if info["MODEL_MODE"] == "Deep_Learning":
        test_dataset = utils.CustomDataset(test_x_s)
        test_dataloader = DataLoader(test_dataset, batch_size=info["BATCH_SIZE"] , shuffle=False)
    
        data = test_dataloader
    
    elif info["MODEL_MODE"] == "Machine_Learning":
        data = test_x_s
    
    return data

def deep_inference(model, test_dataloader, device):
    predict = []
    model.to(device)
    model.eval() # 평가
    with torch.no_grad():
        for X in tqdm(iter(test_dataloader)):
            X = X.to(device)
            output = model(X)
                   
            predict.extend(output.cpu().numpy())

    return np.array(predict)

def machine_inference(model_mode, model, test_x_s):
    if "Xgb" in model_mode:
        dtest = xgb.DMatrix(data=test_x_s)
        predict = model.predict(dtest)
    else:
        predict = model.predict(test_x_s)
    
    return np.array(predict)

# 추론
def inference(info, model, test_data):
    if info["MODEL_MODE"] == "Deep_Learning":
        predict = deep_inference(model, test_data, info["DEVICE"])
    
    elif info["MODEL_MODE"] == "Machine_Learning":
        predict = machine_inference(info["MODEL"], model, test_data)
    
    return predict