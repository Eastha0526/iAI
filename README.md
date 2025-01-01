# iAI (공정 예측 시스템)

## 파일 설명

### config.yaml
  : raw_data들 경로, 하이퍼 파라미터(device, batch_size, epoch, learning rate 등 학습 및 추론에 사용할 파라미터 정의 -> train에서만 수정
  : 모델 이름 : 가장 최근에 사용한 모델 이름으로 자동 업데이트 -> 추론 시 자동으로 업데이트 된 정보 사용
  : 모델 구분 : 딥러닝 모델인지, 머신러닝 모델인지 구별
  : 모델 별 파라미터: 모델 별 사용하는 파라미터 정의

  : 추가로 필요하면 수정하면 됩니다.

### utils
  - data_utils.py
    : 데이터 처리와 관련된 함수 모음
    (파일별 호출 및 밀린 데이터 보정, 병합, idx 생성 및 제거, 이상치 제거, Duration_100 scale, 인코딩, 데이터 분할, 스케일, 전처리 통합)

  - train_utils.py
    : 학습을 위한 함수 모음
    (학습을 위한 데이터 구조, 조기 종료, 딥러닝 학습, 딥러닝 검정, 머신러닝 학습 및 검정, 훈련 및 평가 함수)
    훈련 및 평가 함수: info["MODEL_MODE"]에 따라 딥러닝 학습 및 검정과 머신러닝 학습 및 검정으로 나뉘어 시행
  
  - inference_utils.py
    : 추론을 위한 함수 모음
    (전처리에 필요한 encoder 및 scaler 호출, 추론을 위한 데이터 구조, 딥러닝 추론, 머신러닝 추론, 추론 함수)
    추론 함수: info["MODEL_MODE"]에 따라 딥러닝 추론 및 머신러닝 추론으로 나뉘어 시행

  - utils.py
    : 데이터 처리, 학습 및 검정, 추론 이외에 활용되는 함수 모음
    (깔끔한 도움말 출력, yaml 파일 덮어쓰기, Custom Dataset)

### models.py
  : 모델 클래스 모음

### train.py
  : pyyaml이 아닌 ruamel.yaml 사용
  : 입력값 및 모델에 따른 yaml 파일 업데이터 (덮어쓰기 및 저장)
  : 입력으로 모델 이름을 받는다.(필수 X) - 미입력 시 가장 최근에 저장된 model 이름이 사용된다.
  : 덮어쓴 config 파일 저장, raw_data 전처리, encoder, scaler 저장, 학습 및 모델 가중치 저장까지 실행

### inference.py
  : 추론 코드, 저장된 encoder, scaler, model 파일을 활용하여 입력받은 값을 추론하는 데 사용.
  : 입력 받은 값의 갯수, type 별 갯수 점검, Duration_100 추가( 100개의 데이터 생성 - 0.0 ~ 99.0 ), 추론 및 예측 값 표시

---

### 파일 구조

- encoder : 학습된 encoder 저장 공간
  - RPM_label_encoder.pkl
  - Temp_label_encoder.pkl

- raw_data
  - rpm 데이터.csv
  - temp 데이터.csv
  - bulk 데이터.csv
  - proc 데이터.csv

- model_weight : 학습된 모델 가중치 저장 공간
  - RPM_model.pth 
  - Temp_model.pth

- scaler : 학습된 scaler 저장 공간
  - RPM_label_minmax_scaler.pkl
  - Temp_label_minmax_scaler.pkl

utils : utils 함수들 저장 공간
  - data_utils.py
  - train_utils.py
  - inference_utils.py
  - utils.py

- config.yaml
- inference.py
- models.py
- train.py

---

### 실행 환경

```bash
python -m venv venv
source venv/bin/activate
```

## 실행 코드
### 도움말
python [train.py | inference.py] -h : 별칭, 헤더, 설명
예시:
python train.py -h

### 모델 업데이트 및 가중치 저장
python train.py -model(필수 X) 모델 클래스 이름
예시: (Linear, TimeSeriesModel, XgbOptuna, RandomForestRegressorOptuna)
python train.py -model Linear

### 추론
python inference.py -keyword(필수) RPM인지 Temp인지 -input(필수) 42개의 입력값, 따옴표 붙여서 입력 필수, 리스트로 입력받는다. (Duration_100은 내부에서 추가, 우선은 임의로 지정)
예시:
python inference.py -keyword RPM -input "스킨" "메인믹서 혼합2" "메인믹서 혼합" "0.0" "0.0" "0.0" "0.0" "0.0" "1.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "1." "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "1.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "1200" "2000"
python inference.py -keyword Temp -input "톤업크림" "유상 혼합" "유상 혼합" "0.0" "0.0" "0.0" "1.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "1.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "1.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "0.0" "1800" "2000"