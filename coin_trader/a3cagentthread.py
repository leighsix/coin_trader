import threading
from combined_data_preprocessing import preprocess_data
from ppoa3cagent import PPOA3CAgent
import pyupbit
import logging
import os

logging.basicConfig(filename='ppoa3c_trading_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# 데이터 수집 함수
def collect_data(coin, interval=15, count=100):
    df = pyupbit.get_ohlcv(coin, interval=f"minute{interval}", count=count)
    if df is None or df.empty:
        logging.error(f"{coin} 데이터 수집 실패: 데이터가 비어있습니다.")
    return df


# A3C 에이전트를 실행하기 위한 스레드 클래스
class A3CAgentThread(threading.Thread):
    def __init__(self, agent, num_episodes, batch_size=32, interval=15, count=100):
        super(A3CAgentThread, self).__init__()
        self.agent = agent
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.interval = interval
        self.count = count

    def run(self):
        print(f"{self.agent.coin} - A3C 에이전트 학습 시작...")
        total_reward = 0
        total_actions = 0
        total_current_value = 0
        total_buy_count = 0
        total_sell_count = 0
        total_hold_count = 0
        df = collect_data(self.agent.coin, self.interval, self.count)

        for episode in range(self.num_episodes):
            print(f"{self.agent.coin} - Episode {episode + 1}/{self.num_episodes}")
            self.agent.capital = 1000000  # 초기 자본
            self.agent.balance = 0
            self.agent.current_value = 1000000  # 현재 자본 가치

            epoch_reward = 0
            actions_taken = 0
            buy_count = 0
            sell_count = 0
            hold_count = 0

            # 에이전트를 학습 모드로 설정
            self.agent.set_training_mode(mode=True)

            # 초기 상태 설정
            preprocessed_df = preprocess_data(df)


            state = self.agent.get_initial_state(preprocessed_df)  # 초기 상태 가져오기
            for i in range(len(preprocessed_df) - 2):
                # 행동 선택

                action = self.agent.act(state)  # 행동 선택

                # 거래 실행
                current_price = df.iloc[i + 1]['close']
                next_price = df.iloc[i + 2]['close']
                next_state, reward, done = self.agent.step(preprocessed_df, state, action, current_price, next_price)  # 거래 실행

                # 보상 및 수익률 계산
                if action == 0:  # 매수 시도
                    actions_taken += 1
                    buy_count += 1
                elif action == 1:  # 매도 시도
                    actions_taken += 1
                    sell_count += 1
                else:
                    hold_count += 1

                # 보상 업데이트
                epoch_reward += reward

                # 에이전트의 경험 저장
                self.agent.remember(state, action, reward, next_state, done)

                # 경험 훈련시키기
                self.agent.train(self.batch_size)  # 배치 크기만큼 훈련

                # 상태 업데이트
                state = next_state  # 다음 상태로 업데이트

            # 에피소드 결과 출력
            print(f"{self.agent.coin} - Episode {episode + 1} - Total Reward: {epoch_reward:.4f}, "
                  f"Actions Taken: {actions_taken}, buy_count:{buy_count}, sell_count:{sell_count}, hold_count:{hold_count}, "
                  f"Final capital = {self.agent.current_value:.2f}")
            total_reward += epoch_reward
            total_actions += actions_taken
            total_current_value += agent.current_value
            total_buy_count += buy_count
            total_sell_count += sell_count
            total_hold_count += hold_count
            agent.save("combined_models/ppo_model.h5")
            # 하이퍼파라미터 조정 예시 (여기서 필요에 따라 조정)
            if episode % 100 == 0:  # 매 100 에피소드마다 하이퍼파라미터 조정
                new_gamma = 0.95 if self.agent.gamma > 0.9 else self.agent.gamma
                new_epsilon = 0.1 if self.agent.epsilon > 0.1 else self.agent.epsilon
                new_learning_rate = 0.00005 if self.agent.optimizer.param_groups[0]['lr'] > 0.00001 else \
                    self.agent.optimizer.param_groups[0]['lr']
                self.agent.adjust_hyperparameters(gamma=new_gamma, epsilon=new_epsilon, learning_rate=new_learning_rate)
        average_value = total_current_value/self.num_episodes
        # 최종 결과 출력
        print(f"{self.agent.coin} - 전체 보상: {total_reward:.4f}, 총 행동 수: {total_actions}, 총 매수:{total_buy_count}, "
              f"총 매도: {total_sell_count}, 총 유지: {total_hold_count} ",
              f"Average final capital over {self.num_episodes} episodes: {average_value:.2f}")
        agent.save("combined_models/ppo_model.h5")


if __name__ == "__main__":
    coin = "KRW-DOGE"
    agent = PPOA3CAgent(coin)
    model_path = 'combined_models/ppo_model.h5'  # 모델 파일 경로
    # 모델 저장 디렉토리 확인 및 생성
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"{model_dir} 디렉토리를 생성했습니다.")

    # 저장된 모델이 있는지 확인
    if os.path.exists(model_path):
        print(f"{model_path} 파일이 발견되었습니다. 기존 모델을 불러옵니다.")
        agent.load(model_path)  # 저장된 모델 불러오기
    else:
        print(f"{model_path} 파일이 발견되지 않았습니다. 새롭게 학습을 시작합니다.")
    a3c_thread = A3CAgentThread(agent, num_episodes=100, interval=15, count=200)
    a3c_thread.start()
