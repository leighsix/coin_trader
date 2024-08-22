import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import os
import logging
from collections import deque
from coin_model import CoinModel


class PPOA3CAgent:
    def __init__(self, coin, state_size=16, action_size=3, gamma=0.99, epsilon=0.2, epsilon_greedy=0.1,
                 learning_rate=0.0001, initial_capital=1000000, initial_balance=0):
        self.coin = coin
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # 경험 저장을 위한 메모리
        self.gamma = gamma  # 할인율
        self.epsilon = epsilon  # PPO 클리핑을 위한 epsilon
        self.epsilon_greedy = epsilon_greedy  # ε-greedy 정책을 위한 epsilon
        self.model = CoinModel(state_size=state_size, action_size=action_size)  # CoinModel 인스턴스 생성
        self.optimizer = optim.Adam(self.model.model.parameters(), lr=learning_rate)  # 옵티마이저
        self.criterion = nn.MSELoss()  # 손실 함수
        self.training_mode = True  # 학습 모드 플래그
        self.capital = initial_capital  # 초기 자본
        self.balance = initial_balance
        self.current_value = initial_capital  # 현재 자본 가치

    def set_training_mode(self, mode):
        """학습 모드 설정"""
        self.training_mode = mode
        if mode:
            self.model.model.train()  # 학습 모드로 설정
            print("Training mode activated.")
        else:
            self.model.model.eval()  # 평가 모드로 설정
            print("Evaluation mode activated.")

    def remember(self, state, action, reward, next_state, done):
        """경험을 메모리에 저장"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """행동 선택"""
        state_tensor = torch.tensor(state, dtype=torch.float32).to('cuda')  # 상태를 텐서로 변환
        state_tensor = state_tensor.view(1, 10, self.state_size)  # (batch_size, channels, sequence_length)

        if np.random.rand() < self.epsilon_greedy:
            return np.random.randint(self.action_size)  # 무작위 행동 선택
        else:
            with torch.no_grad():
                dqn_output = self.model.model(state_tensor)  # DQN 출력 계산
                action = np.argmax(dqn_output.cpu().numpy())  # 최대 Q값을 가진 행동 선택
                return action

    def ppo_update(self, states, actions, rewards, next_states, dones):
        """PPO 업데이트"""
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            state_tensor = torch.tensor(state, dtype=torch.float32).to('cuda')  # 상태를 텐서로 변환
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to('cuda')  # 다음 상태를 텐서로 변환

            dqn_output = self.model.model(state_tensor)
            next_dqn_output = self.model.model(next_state_tensor)

            # Advantage 계산
            advantage = reward + (1 - done) * self.gamma * torch.max(next_dqn_output).item() - dqn_output[0][action]

            # 손실 계산
            loss = self.criterion(dqn_output[0][action],
                                  torch.tensor(reward, dtype=torch.float32).to('cuda') + advantage)

            self.optimizer.zero_grad()  # 기울기 초기화
            loss.backward()  # 역전파
            self.optimizer.step()  # 가중치 업데이트

    def train(self, batch_size=32):
        """에이전트를 학습시키는 메서드"""
        if len(self.memory) < batch_size:
            return  # 메모리에 충분한 경험이 없으면 학습하지 않음
        minibatch = random.sample(self.memory, batch_size)  # 랜덤 샘플링
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # PPO 업데이트
        self.ppo_update(states, actions, rewards, next_states, dones)

    def save(self, filename):
        """모델 저장"""
        with h5py.File(filename, 'w') as f:
            # 모델 구조를 NumPy 배열로 변환하여 저장
            model_structure = self.model.model.state_dict()  # 모델 구조 가져오기
            # 모델 구조를 저장할 수 있는 형식으로 변환
            for key, value in model_structure.items():
                f.create_dataset(key, data=value.cpu().numpy())  # 텐서를 NumPy 배열로 변환하여 저장

        print("PPO-A3C 모델이 .h5 형식으로 저장되었습니다.")

    def load(self, filename):
        """모델 로드"""
        if os.path.exists(filename):
            with h5py.File(filename, 'r') as f:
                # 모델 가중치 로드
                state_dict = {key: torch.tensor(f[key][()]) for key in f.keys()}
                self.model.model.load_state_dict(state_dict, strict=False)
                self.model.model.eval()  # 평가 모드로 전환
            print("PPO-A3C 모델이 .h5 형식으로 로드되었습니다.")
        else:
            logging.error(f"{filename} PPO-A3C 모델이 존재하지 않습니다.")

    def step(self, preprocessed_df, state, action, current_price, next_price):
        """주어진 행동에 따라 환경에서 다음 상태와 보상을 반환하는 메서드"""
        # 다음 상태, 보상, 종료 여부를 계산하는 로직을 추가해야 함
        next_state_data = self.get_next_state_data(preprocessed_df, state, action, current_price, next_price)
        next_state = next_state_data['state']  # 다음 상태
        reward = next_state_data['reward']  # 보상
        done = next_state_data['done']  # 종료 여부
        return next_state, reward, done  # 다음 상태, 보상, 종료 여부 반환

    def update_capital(self, action, current_price, next_price):
        # 매수, 매도, 유지 결정
        if action == 0:  # 매수
            amount_to_buy = self.capital * 0.3  # 30% 매수
            if amount_to_buy > 5000:  # 매수할 금액이 있는 경우
                # 매수 주문 실행
                self.capital = self.capital - amount_to_buy * 1.0005  # 30% 매수
                self.balance = self.balance + (amount_to_buy / current_price)
                self.current_value = self.capital + (self.balance * next_price)
        elif action == 1:  # 매도
            if self.balance > 0:  # 매도할 코인이 있는 경우
                # 매도 주문 실행
                self.capital = self.capital + self.balance * current_price * 0.9995
                self.balance = 0
                self.current_value = self.capital
        return self.capital, self.balance, self.current_value

    def adjust_hyperparameters(self, gamma=None, epsilon=None, learning_rate=None):
        """하이퍼파라미터 조정 메서드"""
        if gamma is not None:
            self.gamma = gamma
            print(f"Discount factor (gamma) updated to: {self.gamma}")
        if epsilon is not None:
            self.epsilon = epsilon
            print(f"PPO epsilon updated to: {self.epsilon}")
        if learning_rate is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
            print(f"Learning rate updated to: {learning_rate}")

    def get_initial_state(self, preprocessed_df):
        """초기 상태를 가져오는 메서드"""
        # 초기 상태 설정: 과거 10개의 3분봉 데이터 가져오기
        initial_state = preprocessed_df.iloc[0:10][['close', 'high', 'low', 'volume', 'rsi', 'macd',
                                                    'bollinger_high', 'bollinger_low', 'cci',
                                                    'stochastic_k', 'stochastic_d', 'atr',
                                                    'williams_r', 'momentum', 'vwap', 'ema20']].values

        # (10, 16) 형태로 변환하여 (1, 10, 16) 형태로 변경
        initial_state = initial_state.reshape(1, 10, -1)
        return initial_state  # 초기 상태 반환

    def get_next_state_data(self, preprocessed_df, state, action, current_price, next_price):
        # 현재 상태의 인덱스를 찾기
        current_index = preprocessed_df.index[(preprocessed_df[['close', 'high', 'low', 'volume', 'rsi', 'macd',
                                                                'bollinger_high', 'bollinger_low', 'cci',
                                                                'stochastic_k', 'stochastic_d', 'atr',
                                                                'williams_r', 'momentum', 'vwap', 'ema20']] == state[0, -1]).all(axis=1)].tolist()

        if current_index:
            current_index = current_index[0]
            current_index = preprocessed_df.index.get_loc(current_index)  # 몇 번째인지 계산

            next_index = current_index + 1

            if next_index < len(preprocessed_df) - 9:  # 다음 상태를 가져올 수 있는지 확인
                next_state = preprocessed_df.iloc[next_index:next_index + 10][
                    ['close', 'high', 'low', 'volume', 'rsi', 'macd',
                     'bollinger_high', 'bollinger_low', 'cci',
                     'stochastic_k', 'stochastic_d', 'atr',
                     'williams_r', 'momentum', 'vwap', 'ema20']].values

                # (10, 16) 형태로 변환하여 (1, 10, 16) 형태로 변경
                next_state = next_state.reshape(1, 10, -1)

                reward = (self.update_capital(action, current_price, next_price)[2] - 1000000) / 1000000
                done = (next_index + 10 >= len(preprocessed_df))  # 마지막 상태에서 종료 여부 결정
            else:
                next_state = state  # 마지막 상태일 경우 현재 상태 유지
                reward = (self.update_capital(action, current_price, next_price)[2] - 1000000) / 1000000
                done = True
        else:
            next_state = state
            reward = (self.update_capital(action, current_price, next_price)[2] - 1000000) / 1000000
            done = True

        return {
            'state': next_state,  # 다음 상태
            'reward': reward,  # 보상
            'done': done  # 종료 여부
        }