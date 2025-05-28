import torch
import torch.nn as nn

# CNN + LSTM 혼합 구조의 음성 딥보이스 판별 모델 클래스 정의
class CNNLSTM(nn.Module):
    def __init__(self, input_channels=128):  # 입력 채널 수 = Mel 벡터 수
        super(CNNLSTM, self).__init__()

        # CNN 계층: 1D convolution으로 시간축의 특징 추출
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),  # Conv1D: [128, T] → [64, T]
            nn.ReLU(),
            nn.MaxPool1d(2)  # [64, T] → [64, T/2]
        )

        # LSTM 계층: 시간 정보를 고려한 양방향 LSTM
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=32, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )

        # 최종 이진 분류기: sigmoid 출력 (real/fake)
        self.classifier = nn.Linear(32*2, 1)  # 양방향 LSTM의 출력: 32*2

    def forward(self, x):
        # 입력: [batch, channel, time] → CNN 처리
        x = self.cnn(x)

        # LSTM을 위해 차원 순서 변경: [batch, channel, time] → [batch, time, channel]
        x = x.permute(0, 2, 1)

        # LSTM 적용
        out, _ = self.lstm(x)

        # 마지막 time step의 출력을 사용
        out = out[:, -1, :]

        # 이진 분류 결과 sigmoid 출력
        out = torch.sigmoid(self.classifier(out))
        return out

# 모델 파일(.pt)을 로드하고 평가 모드로 반환하는 함수
def load_model(model_path):
    model = CNNLSTM()  # 모델 인스턴스 생성
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))  # weight 로드
    model.eval()  # 추론 모드로 설정
    return model
