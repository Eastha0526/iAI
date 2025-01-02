import argparse
import numpy as np
import pandas as pd
import joblib
import torch
import xgboost as xgb
from ruamel.yaml import YAML
from charset_normalizer import detect
# ---
from utils import utils, inference_utils
import models

import warnings
warnings.filterwarnings("ignore")

def parse_args(): # 실행 시 입력값 저장
    parser = argparse.ArgumentParser(description='Process input data', formatter_class=utils.DynamicHelpFormatter)
    parser.add_argument("-keyword", "--select_keyword", type=str, required=True, help="Select Keyword : RPM or Temp")
    parser.add_argument("-input" ,"--features", type=str, nargs="+", required=True, help="List of 42 features.") # 모든 입력값을 str로 받고 리스트 처리
    return parser.parse_args()

def process_features(features):
    """
    입력된 문자열 리스트를 float와 str로 분리.
    
    Args:
        features (list[str]): 입력된 특성 리스트.
    
    Returns:
        list[float], list[str]: 숫자와 문자열로 분리된 리스트.
    """
    features_list = []
    for feature in features:
        try:
            features_list.append(float(feature))  # 숫자로 변환 시도
        except ValueError:
            features_list.append(feature)  # 실패 시 문자열로 분류
    
    # 개수 검증
    floats = [feature for feature in features_list if type(feature)==float]
    strings = [feature for feature in features_list if type(feature)==str]
    
    if len(floats) != 39:
        raise ValueError(f"Expected 39 float values, but got {len(floats)}.")
    if len(strings) != 3:
        raise ValueError(f"Expected 3 string values, but got {len(strings)}.")
    
    return features_list

def main(args, info):    
    # 키워드 지정
    keyword = args.select_keyword
    
    # 파일 경로 지정
    model_path = f"./model_weight/{keyword}_model.pth"
    encoder_path = f"./encoder/{keyword}_label_encoder.pkl"
    scaler_path = f"./scaler/{keyword}_label_minmax_scaler.pkl"
    
    
    # 입력받은 값 데이터 형식 변환, x: DataFrame
    features = process_features(args.features)
    cols = ['TypeJH', 'OPRDSC_1', 'OPRDSC_2', 's6AA1', 's6AA4', 's6BB1', 's6BB2',
       's6CA1', 's6CA2', 's6CB1', 's6CB2', 's6CC1', 's6DA1', 's6DA2', 's6EA1',
       's6EA2', 's6F00', 's6HA1', 's6HA2', 's6KA1', 's6KA2', 's6OA0', 's6OB0',
       's6PA1', 's6PA2', 's6SA1', 's6SA2', 's6SA3', 's6SB1', 's6SC1', 's6SD1',
       's6SE1', 's6UA1', 's6UA2', 's6UC1', 's6VA1', 's6VA2', 's6WA1', 's6WA3',
       's6YA4', 'SORCURQ', 'RFMCAP']
    
    ## Duration_100 붙이기
    test_x = pd.DataFrame([features]*100, columns=cols)
    test_x["Duration_100"] = [float(per) for per in range(0, 100)]
    
    # encoder, scaler 호출
    loaded_encoder, loaded_scaler = inference_utils.load_preprocessors(encoder_path, scaler_path)
    
    # Preprocessing
    object_cols = ["TypeJH", "OPRDSC_1", "OPRDSC_2"]
    
    ## encoder
    try:
        for col in object_cols:
            try:
                test_x[col] = loaded_encoder[col].transform(test_x[col])
            except ValueError as e:
                 # 어떤 값이 문제인지 출력
                unknown_classes = set(test_x[col]) - set(loaded_encoder[col].classes_)
                print(f"Error: Column '{col}' contains unknown classes: {unknown_classes}")
                raise ValueError(f"Column '{col}' has classes that were not seen during training: {unknown_classes}")
        
         ## scaler
        test_x[cols] = loaded_scaler.transform(test_x[cols])
        test_x_s = test_x.values
                
    except Exception as e:
        print(f"An error occurred during data processing: {e}")
        raise
    
    # Inference
    ## make test data # dataloader or numpy.ndarray
    test_data = inference_utils.make_inference_model_data(info, test_x_s)
    
    ## loaded model
    model_class = getattr(models, info["MODEL"], None)
    model_params = info.get(info["MODEL"], {})
    loaded_model = model_class(**model_params)
    
    if info["MODEL_MODE"] == "Deep_Learning":
        # state_dict 로드
        state = torch.load(model_path)
        loaded_model.load_state_dict(state)
    
    elif info["MODEL_MODE"] == "Machine_Learning":
        if "Xgb" in info["MODEL"]:
            # model.pth 로드
            loaded_model = xgb.Booster()
            loaded_model.load_model(model_path)
        else:
            loaded_model = joblib.load(model_path)
    
    ## 추론
    predicted_values = inference_utils.inference(info, loaded_model, test_data)
    print(f"\n예측 {keyword}: {np.round(predicted_values, 0)}")

    return str(predicted_values)
    
if __name__ == '__main__': # 추론 파일
    args = parse_args()
    
    # yaml 파일 호출
    yaml = YAML()
    yaml.preserve_quotes = True  # 따옴표 유지
    with open('config.yaml', 'rb') as f:
        raw_data = f.read()
        result = detect(raw_data)
        print(f"Detected encoding: {result['encoding']}")

    with open('config.yaml', 'r', encoding=result['encoding']) as f:
        info = yaml.load(f)
        
    print("program start")
    main(args, info)