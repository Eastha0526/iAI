import argparse
#---
import torch
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore")

# 깔끔한 도움말 출력
# 사용자 정의 HelpFormatter
class DynamicHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_main_option_length = 0
        self.max_alias_length = 0

    # 각 옵션의 최대 길이를 계산하고 저장
    def add_argument(self, action):
        option_strings = action.option_strings
        if option_strings:
            main_option = [opt for opt in option_strings if opt.startswith('--')][0] # main(ex. --help)옵션 추출
            alias = [opt for opt in option_strings if not opt.startswith('--')][0] if len(option_strings) > 1 else "" # 갯수가 1개면 main만 입력된 것이기에 ""처리 (ex. -h)
            self.max_main_option_length = max(self.max_main_option_length, len(main_option)) # 길이 업데이트
            self.max_alias_length = max(self.max_alias_length, len(alias)) # 길이 업데이트
        super().add_argument(action)

    # 옵션과 별칭의 간격을 동적으로 조정
    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)

        main_option = [opt for opt in action.option_strings if opt.startswith('--')][0]
        alias = [opt for opt in action.option_strings if not opt.startswith('--')][0] if len(action.option_strings) > 1 else ""
        return f"{main_option:<{self.max_main_option_length + 2}} {alias:<{self.max_alias_length + 2}}" # 메인과 별칭의 포멧 정의

    def _format_action(self, action):
        # 옵션과 설명을 같은 줄에 배치
        invocation = self._format_action_invocation(action)
        help_text = action.help or ""
        return f"  {invocation} {help_text}\n" # (메인+별칭) + 설명 포멧 정의

# yaml 파일 덮어쓰기
def update_config_with_args(config, args):
    for key, value in vars(args).items():
        if value is not None and key in config:
            config[key] = value
    return config

class CustomDataset(Dataset):
    def __init__(self, x, y=None):
        """
        Args:
            x (np.ndarray): 데이터프레임 형태의 입력 데이터.
            y (np.ndarray or None): 타겟 값. 추론 시 None일 수 있음.
        """
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        x = self.x[index]
        
        if self.y is not None:
            y = self.y[index]
            return x, y
        else:
            return x  # 추론 시에는 x만 반환