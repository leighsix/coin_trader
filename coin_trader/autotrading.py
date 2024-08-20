from ppoa3cagent import PPOA3CAgent
from combined_data_preprocessing import preprocess_data
import logging
import time
import pyupbit
import torch
import threading

# API 키 설정
with open("key_info.txt") as f:
    lines = f.readlines()
    acc_key = lines[0].strip()
    sec_key = lines[1].strip()

# Upbit API 객체 생성
upbit = pyupbit.Upbit(acc_key, sec_key)

# 로깅 설정
logging.basicConfig(filename='cnn_trading_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# 데이터 수집 함수
def collect_data(coin):
    df = pyupbit.get_ohlcv(coin, interval="minute60", count=100)
    if df is None or df.empty:
        logging.error(f"{coin} 데이터 수집 실패: 데이터가 비어있습니다.")
    return df

# 자동매매 클래스 정의
class AutoTrading:
    def __init__(self, agent, coin):
        self.agent = agent  # PPO-A3C 에이전트
        self.coin = coin  # 거래할 코인
        self.buy_price = None  # 구매 가격 초기화
        self.running = True  # 자동매매 실행 플래그

    def get_current_price(self):
        """현재 가격을 가져오는 메서드"""
        return pyupbit.get_current_price(self.coin)  # 현재 가격 반환

    def get_balance(self):
        """현재 보유 자산을 가져오는 메서드"""
        try:
            balance = upbit.get_balance("KRW")  # 원화 잔고 조회
            if balance is None:
                logging.error("잔고 조회 실패: None 반환")
                return 0  # 잔고가 없을 경우 0 반환
            return balance
        except Exception as e:
            logging.error(f"잔고 조회 중 오류 발생: {e}")
            return 0  # 오류 발생 시 0 반환

    def execute_trade(self, action):
        """주어진 행동에 따라 거래를 실행하는 메서드"""
        current_price = self.get_current_price()  # 현재 가격 가져오기
        if action == 0:  # 매수
            print("매수를 시도합니다.")
            krw_balance = self.get_balance()  # 현재 보유 원화 잔고
            amount_to_buy = krw_balance * 0.3  # 30% 매수
            if amount_to_buy > 5000:  # 매수할 금액이 있는 경우
                # 매수 주문 실행
                upbit.buy_market_order(self.coin, amount_to_buy)  # 30% 매수
                self.buy_price = current_price  # 구매 가격 저장
                print(f"Bought {self.coin} at {self.buy_price} KRW")
            else:
                print("매수할 금액이 부족합니다.")
        elif action == 1:  # 매도
            print("매도를 시도합니다.")
            coin_balance = upbit.get_balance(self.coin)  # 보유 코인 잔고 조회
            if coin_balance is not None and coin_balance > 0:  # 매도할 코인이 있는 경우
                # 매도 주문 실행
                upbit.sell_market_order(self.coin, coin_balance)  # 보유 코인 모두 매도
                print(f"Sold {self.coin} at {current_price} KRW")
                self.buy_price = None  # 구매 가격 초기화
            else:
                print("보유코인 잔고가 없습니다.")
        elif action == 2:  # 유지
            print("현상태를 유지하겠습니다.")
            print(f"Holding {self.coin} at {current_price} KRW")

    def run(self):
        """자동매매를 실행하는 메서드"""
        while self.running:
            # 시장 데이터 가져오기
            market_data = collect_data(self.coin)
            market_data = preprocess_data(market_data)
            # 상태를 텐서로 변환
            if market_data is not None:
                state = market_data.iloc[-1][['close', 'high', 'low', 'volume', 'rsi', 'macd', 'bollinger_high',
                                              'bollinger_low', 'cci', 'stochastic_k', 'stochastic_d',
                                              'atr', 'williams_r', 'momentum', 'vwap', 'ema20']].values.reshape(1, -1)
                # 행동 선택
                action = self.agent.act(state)  # 행동 선택
                # 거래 실행
                self.execute_trade(action)  # 선택한 행동에 따라 거래 실행
                time.sleep(1200)  # 20분 대기 (20분마다 거래 결정)
            else:
                print(f"{self.coin} 데이터가 없습니다.")
                time.sleep(60)
    def stop(self):
        """자동매매 중지 메서드"""
        self.running = False
        print("Auto trading stopped.")


def auto_trade(agent, coin):
    auto_trader = AutoTrading(agent, coin)
    try:
        auto_trader.run()  # 자동매매 실행
    except KeyboardInterrupt:
        auto_trader.stop()  # Ctrl+C로 중지 시 자동매매 중지


# 메인 실행 부분
if __name__ == "__main__":
    # 자동매매 클래스 초기화
    coins = ["KRW-BTC", "KRW-ETH", "KRW-SOL", "KRW-XRP", "KRW-QKC", "KRW-DOGE", "KRW-AAVE"]  # 거래할 코인
    model_path = 'combined_models/ppo_model.h5'  # 모델 파일 경로
    threads = []
    for coin in coins:
        agent = PPOA3CAgent(coin)
        agent.load(model_path)  # 저장된 모델 로드
        agent.set_training_mode(mode=False)
        thread = threading.Thread(target=auto_trade, args=(agent, coin))
        thread.start()
        threads.append(thread)
        time.sleep(3)  # 각 스레드 시작 사이에 대기

        # 모든 스레드 종료 대기
    for thread in threads:
        thread.join()

    logging.info(f"{coins} 자동매매 종료")
    print(f"{coins} 자동매매 종료")
