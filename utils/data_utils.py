import os
import logging
import numpy as np
import pandas as pd
import pickle
import csv
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# 데이터 전처리 함수
def preprocess_dataframe(df):
    logging.info("데이터 전처리 시작")
    # 마지막 5개의 불필요한 행 제거
    df = df[:-5]

    # 밀려난 값 조정
    none_cnt = list(df.iloc[0, :]).count(None)  # 열 목록 중 None인 값의 갯수 == 밀려난 열의 수
    if none_cnt > 0:
        modify_idx = df[df.iloc[:, -1:].notnull().any(axis=1)].index
        for idx in modify_idx:
            # 한 행에서 Null 값의 갯수만큼 col 인덱스 위치 조정
            none_idx = [col_idx for col_idx, value in enumerate(df.iloc[idx, :]) if value == None][:none_cnt]  # 밀려난 만큼만 확보
            raw_idx = [col_idx for col_idx, _ in enumerate(df.iloc[idx, :]) if col_idx not in none_idx]  # 밀린 부분의 열 번호 제외
            raw_idx.extend(none_idx)
            df.iloc[idx, :] = df.iloc[idx, raw_idx]  # 열 순서 조정

    # 조정 후 마지막 열 제거 및 columns name 재설정
    df.columns = [col.strip() if type(col) == "str" else col for col in df.iloc[0, :]]  # None 값이면 그대로 아니면 앞/뒤 공백 제거해서
    col_cnt = len(df.iloc[0, :]) - none_cnt
    df = df.iloc[1:, :col_cnt]

    # 특정 자료형 형식 변경
    if "SEQKEY" in df.columns:
        df["SEQKEY"] = df["SEQKEY"].apply(pd.to_numeric, errors='coerce')

    return df

# 데이터 기입 오류 정정을 위해 csv 파일을 다시 생성
def load_raw_data(file_path):
    csv_list = list()
    # 정상적인 파일 읽기 확인
    try:
        with open(file_path, "rt", encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for l in reader:
                l = [None if value in ["", "NULL"] else value.strip() for value in l]  # null이 아닌 '' 값이 존재하여 해당 값들도 null로 변경, 앞뒤 공백 제거
                csv_list.append(l)
    except FileNotFoundError as e:
        print(f"Error: File not found. Details: {e}")
        print(f"{file_path.split('/')[-1]} 데이터 로드 실패")
        raise

    df = pd.DataFrame(csv_list)
    df = df[:-5] # 확인 결과 마지막 5개의 불필요한 행 제거

    ## 밀려난 값 조정
    none_cnt = list(df.iloc[0, :]).count(None) # 열 목록 중 None인 값의 갯수 == 밀려난 열의 수
    if none_cnt > 0:
        modify_idx = df[df.iloc[:, -1:].notnull().any(axis=1)].index
        # df = swap_null(df, modify_idx, none_cnt)
        for idx in modify_idx:
        # 한 행에서 Null 값의 갯수마큼 col 인덱스 위치 조정
            none_idx = [col_idx for col_idx, value in enumerate(df.iloc[idx, :]) if value==None][:none_cnt] # 밀려난 만큼만 확보
            raw_idx = [col_idx for col_idx, _ in enumerate(df.iloc[idx, :]) if col_idx not in none_idx] # 밀린 부분의 열 번호 제외 
            raw_idx.extend(none_idx) 
            df.iloc[idx, :] = df.iloc[idx, raw_idx] # 열 순서 조정
    
    ## 조정 후 마지막 열 제거 및 columns name 재설정
    df.columns = [col.strip() if type(col)=="str" else col for col in df.iloc[0, :]] # None 값이면 그대로 아니면 앞/뒤 공백 제거해서
    col_cnt = len(df.iloc[0, :]) - none_cnt
    df = df.iloc[1:, :col_cnt]
    
    ## 특정 자료형 형식변경
    df["SEQKEY"] = df["SEQKEY"].apply(pd.to_numeric)
    
    return df

# bulk를 proc에 병합하는 함수
def bulk_to_proc(bulk, proc):
    # bulk 정리
    bulk_cols = list(bulk.iloc[:, 6:].columns) # 첨가 여부 + 현재 진행 용량 + 최대 가능 용량
    bulk_cols.extend(["SORNUMB", "SORITEM"]) # 병합 기준
    bulk_sub = bulk[bulk_cols]
    
    # proc 정리
    proc_cols = list(proc.iloc[:, 6:].columns) # proc의 경우 추후 hmrpm과 mtemp에 SORNUMB와 SEQKEY를 기준으로 병합 -> 이 외의 경우 불필요
    proc_sub = proc.drop(proc_cols, axis=1)
    
    # 병합
    btp = proc_sub.merge(bulk_sub, how="left", on=["SORNUMB", "SORITEM"])
    return btp

# 병합한 btp를 rpm과 temp 데이터에 병합하는 함수
def data_to_btp(raw_data, btp):
    # raw_data 정리
    # ID부터 Time_Act까지 Duration_100 빼고 제거
    delete_cols = ["ID", "ID_Min", "ID_Max", "Time_Act"]
    raw_data_sub = raw_data.drop(delete_cols, axis=1)
    
    # raw_data와 btp 병합
    raw_data_df = raw_data_sub.merge(btp, how="left", on=["SORNUMB", "SEQKEY", "OPRDSC_1", "OPRDSC_2"])
    return raw_data_df

# 불필요 특성을 제거하고, 특성별 적절한 자료형을 지정하는 함수
def change_col_types(raw_data, object_cols):
    # raw_data 정리
    # 병합에 사용된 열 중 불필요 열 제거
    drop_cols = ["SORNUMB", "SORITEM", "SEQKEY"] # 
    df = raw_data.drop(drop_cols, axis=1)
    
    # 데이터 값에 적합한 자료형 부여
    numeric_cols = list(set(df)-set(object_cols))
    df[numeric_cols] = df[numeric_cols].astype("float")

    return df

# 필수 특성의 null값을 제거하고 나머지 특성의 Null값을 치환하는 함수 + OPRDSC_2의 '삭제 예정' 행 제거
def change_null(df, object_cols):
    # OPRDSC_2의 '삭제 예정' 항목 지우기
    drop_df = df[df["OPRDSC_2"]=="삭제 예정"]
    df = df.drop(index=drop_df.index).reset_index(drop=True)
    
    not_null_cols = ["TypeJH", "Duration_100", "SORCURQ", "RFMCAP", "HomoRPM", "MainTemp"] # null이 있으면 제거해야 하는 값
    check_cols = [col for col in not_null_cols if col in df.columns] # columns 중 not_null_cols에 해당되는 열 추출
    df = df.dropna(subset=check_cols, axis=0) # 해당 열의 요소가 0이면 행 전체를 제거
    numeric_cols = [col for col in df.select_dtypes(include="float").columns if col not in not_null_cols] # numeric_cols 정의
    
    null_cols = [col for col in df.columns if df[col].isnull().sum() > 0] # 나머지에서 null이 있는 열 추출
    for col in null_cols:
        if col in object_cols:
            df[col] = df[col].fillna("None") # object는 "None"으로
        elif col in numeric_cols: 
            df[col] = df[col].fillna(0) # numeric은 추가 요소 열만 있었기에 0으로 - 그 이외의 값이 Null이면 행을 제거(TypeJH, HomoRPM, )
    
    return df

# 최종 idx 열 생성 및 df 열 위치 조정
def make_idx(df, object_cols):
    not_idx_cols = ["Duration_100", "HomoRPM", "MainTemp"]
    df_numeric_cols = list(df.select_dtypes(exclude="object").columns)
    df_cols = object_cols + df_numeric_cols
    idx_cols = [col for col in df_cols if col not in not_idx_cols] # idx가 되는 열 모음
    
    # 보기 편한 열 순서
    new_order = [col for col in idx_cols if col in df.columns] + [col for col in df.columns if col not in idx_cols] # idx 열 모으고 마지막 열에 y값 지정
    df = df[new_order]
    
    # 새로운 열 생성
    df["idx"] = ""
    for i, col in enumerate(idx_cols):
        if col in df_numeric_cols:
            df["idx"] += df[col].astype(str)
        else:
            df["idx"] += df[col]
            
        if i < (len(idx_cols)-1):
            df["idx"] += "_"
            
    return df

# unique별 이상치 제거
def iqr_outlier(df):
    df = df.copy() # 원본 보호
    to_remove = []  # 제거할 인덱스 저장용 리스트
    column = df.columns[-2]
    
    # 음수 보정(HomoRPM과 MainTemp가 음수인 경우 0으로 변환)
    minus_outliers = (df[column] < 0)
    df.loc[df.index[minus_outliers], column] = 0
    
    for idx in df["idx"].unique():
        sub = df[df["idx"]==idx] # idx 기준으로 추린 후 이상치 추출
        
        q25, q75 = np.quantile(sub[column], 0.25), np.quantile(sub[column], 0.75)
        iqr = q75 - q25
    
        # cut_off 계산
        cut_off = iqr*1.5
        
        # lower, upper
        lower, upper =  max(0, q25 - cut_off), max(0, q75 + cut_off)
        
        # 이상치 index 모음
        sub_outlier = (sub[column] < lower) | (sub[column] > upper)
        
        # 제거 대상 인덱스 추가
        to_remove.extend(sub.index[sub_outlier])
        
    # 이상치 제거 및 idx 제거
    df = df.drop(index=to_remove).reset_index(drop=True)
    df = df.drop("idx", axis=1)
    
    return df

# 비율 데이터 전처리 및 불필요 열 제거
def time_series_scale(df):
    df = df.copy()
    # Duration_100 전처리
    df["Duration_100"] = df["Duration_100"] / 100
    
    return df

# encoder
def encoding(df, object_cols, save_path):
    encoder_dict = {}
    for col in object_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoder_dict[col] = encoder
    
    # 3. 저장
    pickle.dump(encoder_dict, open(f'{save_path}', 'wb'))
    
    return df

def split(df):
    # 데이터 분리
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True)
    train_df, valid_df = train_test_split(train_df, test_size=0.2, shuffle=True)
    
    # x, y 나누기
    train_df_x, train_df_y = train_df.iloc[:, :-1], train_df.iloc[: , -1]
    valid_df_x, valid_df_y = valid_df.iloc[:, :-1], valid_df.iloc[: , -1]
    test_df_x, test_df_y = test_df.iloc[:, :-1], test_df.iloc[: , -1]
    
    return train_df_x, train_df_y.values, valid_df_x, valid_df_y.values, test_df_x, test_df_y.values

# scaler
def scaling(train_x, valid_x, test_x, save_path):
    scaler = MinMaxScaler()
    exclude_col = ["Duration_100"]
    scale_cols = [col for col in train_x.columns if col not in exclude_col]

    # 나머지 스케일
    train_x[scale_cols] = scaler.fit_transform(train_x[scale_cols])
    valid_x[scale_cols] = scaler.transform(valid_x[scale_cols])
    test_x[scale_cols] = scaler.transform(test_x[scale_cols])
    
    # 저장
    pickle.dump(scaler, open(f'{save_path}', 'wb'))

    return train_x.values, valid_x.values, test_x.values

# data preprocessing
def preprocessing(data, btp, keyword): 
    ## data와 btp 병합
    df = data_to_btp(data, btp)
    print("데이터 병합 완")
    
    ## 데이터 정리 및 자료형 변환
    object_cols = ["TypeJH", "OPRDSC_1", "OPRDSC_2"]
    df = change_col_types(df, object_cols)
    print("열 형식 변환 완")
    
    ## null값 처리
    df = change_null(df, object_cols)
    print("null값 처리 완")
    
    ## 최종 idx 열 생성
    df = make_idx(df, object_cols)
    print("idx 생성 완")
    
    ## idx 열 기준 이상치 제거 및 idx 열 제거
    df = iqr_outlier(df)
    print("이상치 제거 완")
    
    ## Duration_100 데이터 scale
    df = time_series_scale(df)
    
    ## Label Encoding
    encoding_df = encoding(df, object_cols, os.path.join(f"./encoder/{keyword}_label_encoder.pkl"))
    print("인코딩 완")

    ## train, valid, test split
    train_x, train_y, valid_x, valid_y, test_x, test_y = split(encoding_df)
    print("데이터 분할 완")
    
    ## scaling
    train_x_s, valid_x_s, test_x_s = scaling(train_x, valid_x, test_x, os.path.join(f"./scaler/{keyword}_label_minmax_scaler.pkl"))
    print("데이터 Scale 완")
    
    return train_x_s, train_y, valid_x_s, valid_y, test_x_s, test_y
