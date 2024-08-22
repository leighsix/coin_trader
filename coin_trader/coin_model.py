import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# TensorFlow의 로그 레벨을 설정하여 경고 메시지를 숨깁니다.
tf.get_logger().setLevel('ERROR')

class CoinModel:
    def __init__(self, state_size=16, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(10, self.state_size)))  # input_shape 수정
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # Q-value 출력
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict(self, state):
        # 입력 데이터의 형태 확인
        print("Input shape for prediction:", state.shape)
        return self.model.predict(state)

    def fit(self, state, target):
        self.model.fit(state, target, epochs=1, verbose=0)

    def save_model(self, filepath):
        self.model.save(filepath)  # 모델 저장

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)  # 모델 불러오기

