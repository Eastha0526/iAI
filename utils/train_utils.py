import os
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

#---
from utils import utils, inference_utils

# 훈련 모델에 필요한 데이터 구성
def make_train_model_data(info, train_x_s, train_y, valid_x_s, valid_y, test_x_s, test_y):
    if info["MODEL_MODE"] == "Deep_Learning":
        train_dataset = utils.CustomDataset(train_x_s, train_y)
        valid_dataset = utils.CustomDataset(valid_x_s, valid_y)

        train_dataloader = DataLoader(train_dataset, batch_size=info["BATCH_SIZE"] , shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=info["BATCH_SIZE"] , shuffle=False)
        
        data_tuple = (train_dataloader, valid_dataloader, test_x_s, test_y)
    
    elif info["MODEL_MODE"] == "Machine_Learning":
        data_tuple = (train_x_s, train_y, valid_x_s, valid_y, test_x_s, test_y)
    
    return data_tuple

# 딥러닝 조기 종료
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, path='checkpoint.ckpt', delta=0, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.path = path
        self.delta = delta
        self.trace_func = trace_func
        
        self.counter = 0 # 
        self.best_score = None
        self.early_stop = False
        self.val_loss = np.Inf
        self.best_model_state_dict = None
        self.best_epoch = None

    def __call__(self, val_loss, model): # 

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
            
        self.val_loss = val_loss
        self.best_model_state_dict = model.state_dict() # 가중치만 저장
        torch.save(self.best_model_state_dict, os.path.join(self.path)) # 가중치만 저장

def deep_train(model, early_stopper, loss_function, optimizer, train_dataloader, val_dataloader, device, max_epoch):
    model.to(device)
    criterion = loss_function.to(device)
    best_loss = np.inf

    for epoch in range(1, max_epoch+1):
        model.train() # 훈련 상황
        train_loss = []

        for X, Y in tqdm(iter(train_dataloader)):
            X = X.to(device)
            Y = Y.to(device)

            # Foward
            optimizer.zero_grad() # 가중치 초기화

            # get prediction
            output = model(X)

            loss = criterion(output, Y)

            # back propagation
            loss.backward()

            optimizer.step() # 반영

            # Mini Batch 별 평가지표 저장
            train_loss.append(loss.item()) # Loss
        
        # validation 진행
        val_loss = deep_validation(model, val_dataloader, criterion, device)
        
        # 1 Epoch 후 출력
        print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}])')

        # check model early stopping point & save model if the model reached the best performance.
        early_stopper(val_loss, model)
        if early_stopper.early_stop or epoch == max_epoch: # 조기종료 or 종료 시 호출되는 값들
            best_state_dict = early_stopper.best_model_state_dict # 모델 가중치
            best_loss = - early_stopper.best_score
            
            if early_stopper.early_stop:
                print("Early Stop!!")
                print(f"Loss: {best_loss:.5f}") # 
                break

            else:
                print("Last Epoch")
                print(f"Loss: {best_loss:.5f}") # 
            
    return best_state_dict

def deep_validation(model, val_dataloader, criterion, device):
    model.eval() # 평가 상황
    val_loss = []
    with torch.no_grad(): # 가중치 업데이트 X
        for X, Y in tqdm(iter(val_dataloader)):
            X = X.to(device)
            Y = Y.to(device)

            output = model(X)
            loss = criterion(output, Y)

            # 평가지표 저장
            val_loss.append(loss.item()) # Loss
            
    return np.mean(val_loss)

def machine_train(model, train_x_s, train_y, valid_x_s, valid_y, info): # optuna가 아닌 다른 경우도 추가 가능
    best_model = model.optimize(info, train_x_s, train_y, valid_x_s, valid_y) # 하이퍼파라미터 최적화 실행
    
    return best_model

def train(info, model_class, data_dict, model_weight_path):
    metrics_dict = {}
    
    for keyword in data_dict:
        # Model Initialization        
        try:
            # YAML에서 모델 초기화 인자 로드
            model_params = info.get(info["MODEL"], {})
            print(model_params)
            
            # 모델 초기화
            model = model_class(**model_params)
            print(f"{keyword} {info['MODEL']} 초기화 완료")

        except Exception as e:
            print(f"Unexpected error: {e}")
        except TypeError as e:
                print(f"Error during model initialization: {e}")
        
        # 딥러닝 학습
        if info["MODEL_MODE"] == "Deep_Learning":
            train_dataloader, valid_dataloader, test_x_s, test_y = data_dict[keyword]
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=info["LEARNING_RATE"])
            
            # define EarlyStopping.
            early_stopper = EarlyStopping(patience=5, verbose=True, path=os.path.join(model_weight_path, f"{keyword}_model.pth")) # file_name 변경하기
            
            # Training Loop
            model_state = deep_train(model, early_stopper, criterion, optimizer, train_dataloader, valid_dataloader, info["DEVICE"], info["EPOCHS"])
            
            # Inference
            loaded_model = model_class(**model_params)
            loaded_model.load_state_dict(model_state)
        
        # 머신러닝 학습
        elif info["MODEL_MODE"] == "Machine_Learning":
            train_x_s, train_y, valid_x_s, valid_y, test_x_s, test_y = data_dict[keyword]

            # Training Loop
            best_model = machine_train(model, train_x_s, train_y, valid_x_s, valid_y, info)
            
            if "Xgb" in info["MODEL"]:    
                best_model.save_model(os.path.join(model_weight_path, f"{keyword}_model.pth"))
                
                # Inference
                loaded_model = xgb.Booster()
                loaded_model.load_model(os.path.join(model_weight_path, f"{keyword}_model.pth"))
            
            else:
                joblib.dump(best_model, os.path.join(model_weight_path, f"{keyword}_model.pth"))  # 모델 자체 저장
                
                # Inference
                loaded_model = joblib.load(os.path.join(model_weight_path, f"{keyword}_model.pth"))
        
        # Inference + Evalutation
        test_data = inference_utils.make_inference_model_data(info, test_x_s)
        predicted_values = inference_utils.inference(info, loaded_model, test_data)
        mse = mean_squared_error(test_y, predicted_values)
        mae = mean_absolute_error(test_y, predicted_values)
        
        metrics_dict[keyword] = (mse, mae)
            
    return metrics_dict
            
            