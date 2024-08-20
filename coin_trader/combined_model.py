import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# CombinedModel 클래스 정의
class CombinedModel(nn.Module):
    def __init__(self, input_channels, output_size, lstm_hidden_size, rnn_hidden_size, dropout_rate=0.5):
        super(CombinedModel, self).__init__()
        # CNN 레이어
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)  # 64 노드
        self.bn1 = nn.BatchNorm1d(64)  # 배치 정규화
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)  # 128 노드
        self.bn2 = nn.BatchNorm1d(128)  # 배치 정규화

        # LSTM 레이어
        self.lstm = nn.LSTM(128, lstm_hidden_size, batch_first=True)  # LSTM 노드 수
        self.rnn = nn.RNN(lstm_hidden_size, rnn_hidden_size, batch_first=True)  # RNN 노드 수

        # Dropout 레이어
        self.dropout = nn.Dropout(dropout_rate)  # 과적합 방지

        # DQN을 위한 Fully Connected Layer
        self.fc_dqn = nn.Linear(rnn_hidden_size, output_size)  # DQN 출력

        # 가중치 초기화
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LSTM, nn.RNN)):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_uniform_(param, a=0.1)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        # CNN을 통한 전방향 전파
        x = nn.functional.relu(self.bn1(self.conv1(x)))  # 첫 번째 CNN 레이어
        x = nn.functional.max_pool1d(x, kernel_size=2, stride=2)  # 최대 풀링
        x = nn.functional.relu(self.bn2(self.conv2(x)))  # 두 번째 CNN 레이어
        x = nn.functional.max_pool1d(x, kernel_size=2, stride=2)  # 최대 풀링

        # LSTM 입력을 위해 차원 변경
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, num_features) 형태로 변경

        # LSTM을 통한 전방향 전파
        lstm_out, _ = self.lstm(x)  # LSTM 처리
        lstm_out = lstm_out[:, -1, :]  # 마지막 시퀀스의 출력만 사용

        # RNN을 통한 전방향 전파
        rnn_out, _ = self.rnn(lstm_out.unsqueeze(1))  # RNN 입력을 위해 차원 변경
        rnn_out = rnn_out[:, -1, :]  # 마지막 시퀀스의 출력만 사용

        # Dropout 적용
        rnn_out = self.dropout(rnn_out)  # Dropout 적용

        # DQN 출력
        dqn_output = self.fc_dqn(rnn_out)  # DQN 출력
        return dqn_output

