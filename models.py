import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from tqdm import tqdm

import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout):
        super(Linear, self).__init__()
        # 입력값 정의
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

        # 출력 레이어
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size*2, self.hidden_size*4),
            nn.Dropout(self.dropout),
            )
        
        self.fc2 = nn.Linear(self.hidden_size*4, self.output_size)
        
        # Dropout
        self.drop_out = nn.Dropout(self.dropout)
        # Activation Function
        self.actv = nn.ReLU()

    def forward(self, x):
        # Forward
        out = self.fc1(x)
        out = self.drop_out(out)
        out = self.fc2(out)
        out = self.actv(out)

        return out

class TimeSeriesModel(nn.Module):
    """Time Series model combining temporal and static features."""
    def __init__(self, static_input_dim, hidden_dim, output_dim):
        super(TimeSeriesModel, self).__init__()
        # 1D Conv Layer for time series
        self.conv1d = nn.Sequential( 
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=1)
        )
        # Bi-LSTM
        self.bilstm = nn.LSTM( 
            input_size=16,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        # Static Processing
        self.fc_static = nn.Sequential(
            nn.Linear(static_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU()
        )
        # Cross Attention Layers
        self.ts_to_static_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        self.static_to_ts_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        # Final cross attention fusion
        self.cross_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU()
        )
        # Output
        self.fc_final = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )
    
    # 시계열 데이터의 모양이 batch_size, 1, 1
    def cross_attention(self, query, key, value, attention_layer):
        """Compute cross-attention."""
        attention_weights = attention_layer(query)
        context = torch.bmm(attention_weights.transpose(-2, -1), value) # 행렬곱 수행 : X_static과 rolling_data의 결합
        return context
    
    def forward(self, x): 
        # X_static, rolling_data = x["x_static"], x["x_timeSeries"]
        X_static, rolling_data = x[:, :-1], x[:, -1:].reshape(-1, 1, 1) # rolling_data에 맞게 데이터 형식 변경 
        batch_size = rolling_data.size(0)
        rolling_data = rolling_data.reshape(batch_size, 1, -1)
        conv_out = self.conv1d(rolling_data).permute(0, 2, 1)
        ts_features, _ = self.bilstm(conv_out)
        static_features = self.fc_static(X_static)
        static_seq = static_features.unsqueeze(1).repeat(1, ts_features.size(1), 1)
        ts_attended_static = self.cross_attention(ts_features, static_seq, static_seq, self.ts_to_static_attention).squeeze(1)
        static_attended_ts = self.cross_attention(static_seq, ts_features, ts_features, self.static_to_ts_attention).squeeze(1)
        cross_features = torch.cat([ts_attended_static, static_attended_ts], dim=1)
        fused_features = self.cross_fusion(cross_features)
        output = self.fc_final(fused_features)
        return output

class XgbOptuna:
    def __init__(self, num_boost_round, early_stopping_rounds, n_trials):
        super(XgbOptuna, self).__init__()
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.n_trials = n_trials
        self.study = None
        self.best_params = None
        self.best_model = None
        
    # Objective function for Optuna
    def objective(self, trial):
        # 하이퍼파라미터 검색 공간 정의
        param = {
            "verbosity": 0,
            "objective": "reg:squarederror",
            "tree_method": "gpu_hist" if self.device == "cuda" else "hist",  # GPU 활용, CPU만 사용할 경우 "hist"로 변경  # GPU 추론
            "eval_metric": "rmse",  # 평가 지표
            "eta": trial.suggest_float("eta", 0.001, 0.3, log=True),  # 학습률 # trial.suggest_*: 하이퍼파라미터의 검색 공간을 정의합니다.
            "max_depth": trial.suggest_int("max_depth", 1, 200, step=1),  # 트리 깊이
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10, step=1),  # 최소 가중치
            "gamma": trial.suggest_float("gamma", 0, 1, step=0.1),  # 분할 손실 최소화
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),  # 하위 샘플 비율
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),  # 열 샘플 비율
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),  # L2 정규화
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),  # L1 정규화
            'max_bin' : self.batch_size,
        }
        
        # 조기 종료 설정 및 모델 학습
        evals = [(self.dtrain, "train"), (self.dvalid, "valid")]
        model = xgb.train(
            param,
            self.dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,  # 조건부 조기 종료
            verbose_eval=False,
        )

        # 최적화할 평가 점수 반환 (eval 데이터의 best_score)
        return model.best_score

    # Optuna로 최적의 하이퍼파라미터 탐색
    def optimize(self, info, train_x_s, train_y, valid_x_s, valid_y):
        self.dtrain = xgb.DMatrix(train_x_s, label=train_y)
        self.dvalid = xgb.DMatrix(valid_x_s, label=valid_y)
        self.device = info["DEVICE"]
        self.batch_size = info["BATCH_SIZE"]
        self.study = optuna.create_study(direction="minimize")
        
        # Objective function 래핑 - 수정 중
        with tqdm(total=self.n_trials, desc="Hyperparameter Optimization", unit="trial") as pbar:
            def progress_callback(study, trial):
                pbar.update(1)

            self.study.optimize(lambda trial: self.objective(trial), n_trials=self.n_trials, callbacks=[progress_callback], show_progress_bar=False)
        
        self.best_params = self.study.best_params
        self.best_params.update({
            'objective': 'reg:squarederror',
            'tree_method': "gpu_hist" if self.device == "cuda" else "hist",
            'eval_metric': 'rmse',
            'max_bin': self.batch_size,
        })

        # 최적의 하이퍼파라미터로 모델 재학습
        evals = [(self.dtrain, 'train'), (self.dvalid, 'eval')]
        self.best_model = xgb.train(
            params=self.best_params,
            dtrain=self.dtrain,
            num_boost_round=self.study.best_trial.number + 5, # 50, # 확인을 위해 짧게 설정
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False
        )
        
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call train_best_model() first.")
        
        return self.best_model
    
class RandomForestRegressorOptuna:
    def __init__(self, n_trials):
        super(RandomForestRegressorOptuna, self).__init__()
        self.n_trials = n_trials
        self.study = None
        self.best_params = None
        self.best_model = None
    
    def objective(self, trial):
        # 하이퍼파라미터 검색 공간 정의
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 501, step=5),
            'max_depth': trial.suggest_int('max_depth', 5, 101, step=5),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 11, step=1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 11, step=1),
            'max_features': trial.suggest_categorical("max_features", ['sqrt', 'log2']),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.01, 0.4, log=True),
        }
        
        # 모델 초기화 및 모델 학습
        model = RandomForestRegressor(**param)
        model.fit(self.train_x_s, self.train_y)
        
        # 검정
        valid_pred = model.predict(self.valid_x_s)
        rmse = mean_squared_error(self.valid_y, valid_pred)  # RMSE 반환

        # 최적화할 평가 점수 반환
        return rmse

    # Optuna로 최적의 하이퍼파라미터 탐색
    def optimize(self, info, train_x_s, train_y, valid_x_s, valid_y):
        self.train_x_s = train_x_s
        self.train_y = train_y
        self.valid_x_s = valid_x_s
        self.valid_y = valid_y
        self.study = optuna.create_study(direction="minimize")
        
        # Objective function 래핑
        with tqdm(total=self.n_trials, desc="Hyperparameter Optimization", unit="trial") as pbar:
            def progress_callback(study, trial):
                pbar.update(1)

            self.study.optimize(lambda trial: self.objective(trial), n_trials=self.n_trials, callbacks=[progress_callback], show_progress_bar=False)
        
        print("\nBest parameters found: ", self.study.best_params)
        print("Best score: ", self.study.best_value)
        
        # 최적의 하이퍼파라미터로 모델 재학습
        best_params = self.study.best_params
        # 최적 하이퍼파라미터로 모델 재학습
        best_model = RandomForestRegressor(**best_params, n_jobs=-1)
        best_model.fit(self.train_x_s, self.train_y)
        
        # 최적 모델 반환
        return best_model