# os
import os
import logging
from dotenv import load_dotenv
# handling data
import pandas as pd
import pytds
from charset_normalizer import detect
# train
import torch
# configuration
from ruamel.yaml import YAML
import argparse
# ---
from utils import data_utils, train_utils, utils
import models

import warnings
warnings.filterwarnings("ignore")

def parse_args(): # 실행 시 입력값 저장
    parser = argparse.ArgumentParser(description='Process input data', formatter_class=utils.DynamicHelpFormatter)
    parser.add_argument("-model", "--MODEL", type=str, required=False, help="Select Model : Default: 최근 사용 모델 이름")
    return parser.parse_args()

def main(info):
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )
    print(f"Device: {info['DEVICE']}, 모델 범주: {info['MODEL_MODE']}, 모델 이름: {info['MODEL']}")
    
    # .env 파일 로드
    load_dotenv()

    # SQL 서버 접속 정보
    server = os.getenv('DB_SERVER')
    port = os.getenv('DB_PORT')
    username = os.getenv('DB_USERNAME')
    password = os.getenv('DB_PASSWORD')
    database = os.getenv('DB_DATABASE')
    # 데이터 프레임 저장을 위한 딕셔너리
    dataframes = {}
    # 데이터베이스 연결
    try:
        connection = pytds.connect(server=server, database=database, user=username, password=password, port=port)
        print("데이터베이스 연결 성공")
    except Exception as e:
        print("데이터베이스 연결 실패:", e)
        exit()


    # 경로 지정
    home_dir = os.getcwd()
    os.makedirs(os.path.join(home_dir, "encoder"), exist_ok=True) # encoder 파일 저장 공간 생성
    os.makedirs(os.path.join(home_dir, "scaler"), exist_ok=True) # scaler 파일 저장 공간 생성
    os.makedirs(os.path.join(home_dir, "model_weight"), exist_ok=True) # 모델 가중치 or 모델 파일 저장 공간 생성
    model_weight_path = os.path.join(home_dir, "model_weight") # 모델 저장 경로 정의
    
    # 데이터프레임 생성
    try:
        logging.info("쿼리 실행 중: WO_Dsc_Bulk_V1")
        bulk = pd.read_sql("SELECT * FROM iAI00.dbo.WO_Dsc_Bulk_V1", connection)
        logging.info("WO_Dsc_Bulk_V1 데이터 불러오기 성공")

        logging.info("쿼리 실행 중: WO_Dsc_Proc_V1")
        proc = pd.read_sql("SELECT * FROM iAI00.dbo.WO_Dsc_Proc_V1", connection)
        logging.info("WO_Dsc_Proc_V1 데이터 불러오기 성공")

        logging.info("쿼리 실행 중: WO_CHV_HMRPM_V1")
        rpm = pd.read_sql("SELECT * FROM iAI00.dbo.WO_CHV_HMRPM_V1", connection)
        logging.info("WO_CHV_HMRPM_V1 데이터 불러오기 성공")

        logging.info("쿼리 실행 중: WO_CHV_MTemp_V1")
        temp = pd.read_sql("SELECT * FROM iAI00.dbo.WO_CHV_MTemp_V1", connection)
        logging.info("WO_CHV_MTemp_V1 데이터 불러오기 성공")
    except Exception as e:
        logging.error("데이터 불러오기 실패: %s", e)

    # 연결 닫기
    connection.close()
    logging.info("데이터베이스 연결 닫힘")
    # 데이터 Preprocessing
    ## bulk와 proc 병합 -> btp 생성
    btp = data_utils.bulk_to_proc(bulk, proc) # rpm과 temp에 공통으로 사용되는 병합 데이터
    print("\nRPM 전처리 시작")
    train_rpm_x_s, train_rpm_y, valid_rpm_x_s, valid_rpm_y, test_rpm_x_s, test_rpm_y = data_utils.preprocessing(rpm, btp, "RPM")
    print("RPM 전처리 완료\n\nTemp 전처리 시작")
    train_temp_x_s, train_temp_y, valid_temp_x_s, valid_temp_y, test_temp_x_s, test_temp_y = data_utils.preprocessing(temp, btp, "Temp")
    print("Temp 전처리 완료")
    
    rpm_tuple = train_utils.make_train_model_data(info, train_rpm_x_s, train_rpm_y, valid_rpm_x_s, valid_rpm_y, test_rpm_x_s, test_rpm_y)
    temp_tuple = train_utils.make_train_model_data(info, train_temp_x_s, train_temp_y, valid_temp_x_s, valid_temp_y, test_temp_x_s, test_temp_y)
    
    data_dict = {"RPM": rpm_tuple,   # 딥러닝 모델이면 (train_dataloader, valid_dataloader, test_x_s, test_y)
                 "Temp": temp_tuple} # 머신러닝 모델이면 (train_x_s, train_y, valid_x_s, valid_y, test_x_s, test_y)
    
    # models 파일에서 info["MODEL"]에 해당하는 클래스를 찾음
    try:
        model_class = getattr(models, info["MODEL"], None)
        if model_class is None:
            raise AttributeError(f"Model {info['MODEL']} not found in models module.")
    except AttributeError as e:
        print(f"Error: {e}")
        raise  # 에러를 다시 던져서 상위 코드에서 처리
    
    print("\n학습 시작")
    # 학습 시작
    metrics_dict = train_utils.train(info, model_class, data_dict, model_weight_path) # 평가 지표 반환
    
    for keyword in metrics_dict: # 출력
        mse, mae = metrics_dict[keyword]
        print(f"\n{keyword} 평가 지표\n"+"-"*10)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}\n")
    print("학습 종료")
    
if __name__ == '__main__':
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
    
    # 입력된 값들로 yaml 파일 덮어쓰기
    info = utils.update_config_with_args(info, args)
    
    info["MODEL_MODE"] = "Deep_Learning" if info["MODEL"] in info["Deep_Learning"] else "Machine_Learning" # 모델 범주 지정
    # info["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu' # 자동 설정
    
    # 덮어쓴 yaml 파일 저장
    with open("config.yaml", "w") as f:
        yaml.dump(info, f)
        
    print("program start")
    main(info)