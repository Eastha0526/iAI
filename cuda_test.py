import torch
print(torch.version.cuda)  # CUDA 버전 확인
print(torch.backends.cudnn.version())  # cuDNN 버전 확인
print(torch.cuda.is_available())  # CUDA 사용 가능 여부 확인