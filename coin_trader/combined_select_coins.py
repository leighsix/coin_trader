import pyupbit
import pandas as pd
import numpy as np
import time

# API 키 설정
with open("key_info.txt") as f:
    lines = f.readlines()
    acc_key = lines[0].strip()
    sec_key = lines[1].strip()

# Upbit API 객체 생성
upbit = pyupbit.Upbit(acc_key, sec_key)


def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_top_coins(num_coins=5):
    markets = pyupbit.get_tickers(fiat="KRW")
    coin_data = []
    coin_list = []

    for market in markets:
        # 5분봉 데이터 가져오기
        df = pyupbit.get_ohlcv(market, interval="minute5", count=60)  # 최근 5시간 데이터
        if df is not None and not df.empty:
            # 변동성 계산 (표준편차)
            volatility = df['close'].pct_change().std()
            # 거래량 계산
            volume = df['volume'].sum()
            # RSI 계산
            df['rsi'] = calculate_rsi(df)
            rsi_value = df['rsi'].iloc[-1]  # 가장 최근 RSI 값

            # 조건: 변동성이 크고, RSI가 특정 범위에 있을 때
            if volatility > 0.02 and (rsi_value < 30 or rsi_value > 70):  # 변동성이 크고 과매도/과매수 상태
                coin_data.append((market, volatility, volume, rsi_value))

    # DataFrame으로 변환
    df_coins = pd.DataFrame(coin_data, columns=['ticker', 'volatility', 'volume', 'rsi'])

    # 변동성과 거래량 기준으로 정렬
    df_coins = df_coins.sort_values(by=['volatility', 'volume'], ascending=False)

    # 상위 num_coins 개수 선택
    top_coins = df_coins.head(num_coins)

    # 조건이 충족되지 않을 경우 안정적인 종목 선택
    if top_coins.empty:
        # 변동성이 큰 종목 중에서 안정적인 종목 선택
        for market in markets:
            df = pyupbit.get_ohlcv(market, interval="minute5", count=60)
            if df is not None and not df.empty:
                volatility = df['close'].pct_change().std()
                volume = df['volume'].sum()
                coin_data.append((market, volatility, volume, 0))  # RSI는 0으로 설정
        df_coins = pd.DataFrame(coin_data, columns=['ticker', 'volatility', 'volume', 'rsi'])
        df_coins = df_coins.sort_values(by=['volatility', 'volume'], ascending=False)
        top_coins = df_coins.head(num_coins)

    top_coins = top_coins[['ticker', 'volatility', 'volume', 'rsi']]

    for ticker in top_coins['ticker']:
        coin_list.append(ticker)

    return coin_list


def sell_all_coins(upbit):
    # 보유한 모든 코인 정보 가져오기
    balances = upbit.get_balances()

    for balance in balances:
        ticker = balance['currency']  # 코인 티커
        amount = float(balance['balance'])  # 보유 수량

        if amount > 0:  # 보유 수량이 0보다 클 경우에만 매도
            market_ticker = "KRW-" + ticker    # 매도 시 사용할 티커
            try:
                # 매도 주문 실행
                sell_result = upbit.sell_market_order(market_ticker, amount)
                print(f"{market_ticker} 매도 주문 성공: {sell_result}")
            except Exception as e:
                print(f"{market_ticker} 매도 주문 실패: {e}")
        time.sleep(2)


def sell_coins(upbit, coin):
    coin_balance = upbit.get_balance(coin)
    if coin_balance > 0:
        amount_to_sell = coin_balance
        upbit.sell_market_order(coin, amount_to_sell)
    else:
        print(f"{coin} 매도할 코인이 없습니다.")
    time.sleep(2)


# 예시 사용
if __name__ == "__main__":
    top_coins = get_top_coins(5)
    print("선택된 코인:", top_coins)