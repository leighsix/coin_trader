import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def preprocess_data(df):
    if df is None or df.empty:
        return None  # None 반환

    # 결측값 처리
    imputer = SimpleImputer(strategy='mean')
    df[['close', 'high', 'low', 'volume']] = imputer.fit_transform(df[['close', 'high', 'low', 'volume']])

    # 기술적 지표 계산
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['rsi'] = compute_rsi(df['close'], 14)
    df['macd'], _ = compute_macd(df['close'])
    df['bollinger_high'] = df['ma20'] + (df['close'].rolling(window=20).std() * 2)
    df['bollinger_low'] = df['ma20'] - (df['close'].rolling(window=20).std() * 2)
    df['cci'] = compute_cci(df)  # CCI 추가
    df['stochastic_k'], df['stochastic_d'] = compute_stochastic(df['high'], df['low'], df['close'])
    df['atr'] = compute_atr(df)
    df['williams_r'] = compute_williams_r(df['high'], df['low'], df['close'])  # Williams %R 추가
    df['momentum'] = compute_momentum(df['close'], 10)  # Momentum 추가
    df['vwap'] = compute_vwap(df)  # VWAP 추가
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()  # EMA 추가
    df['returns'] = df['close'].pct_change().shift(-1)

    # 상승 여부 판단: 0.05% 상승 시 상승으로 판단
    df['target'] = np.where(df['returns'] > 0.0005, 1, 0)  # 0.05% 상승 시 1, 나머지는 0

    # 필요한 열만 선택
    df = df[['close', 'high', 'low', 'volume', 'rsi', 'macd', 'bollinger_high', 'bollinger_low',
             'cci', 'stochastic_k', 'stochastic_d', 'atr', 'williams_r', 'momentum', 'vwap', 'ema20', 'target']]

    # 결측치가 있는 행 제거
    df = df.dropna()
    if df.shape[0] == 0:
        raise ValueError("결측치 제거 후 데이터가 없습니다. 다른 방법으로 결측치를 처리하세요.")
    # 데이터 정규화 (0과 1 사이로)
    df[['close', 'high', 'low', 'volume', 'rsi', 'macd', 'bollinger_high', 'bollinger_low',
        'cci', 'stochastic_k', 'stochastic_d', 'atr', 'williams_r', 'momentum', 'vwap', 'ema20']] = (
        df[['close', 'high', 'low', 'volume', 'rsi', 'macd', 'bollinger_high', 'bollinger_low',
            'cci', 'stochastic_k', 'stochastic_d', 'atr', 'williams_r', 'momentum', 'vwap', 'ema20']] -
        df[['close', 'high', 'low', 'volume', 'rsi', 'macd', 'bollinger_high', 'bollinger_low',
            'cci', 'stochastic_k', 'stochastic_d', 'atr', 'williams_r', 'momentum', 'vwap', 'ema20']].min()) / (
        df[['close', 'high', 'low', 'volume', 'rsi', 'macd', 'bollinger_high', 'bollinger_low',
            'cci', 'stochastic_k', 'stochastic_d', 'atr', 'williams_r', 'momentum', 'vwap', 'ema20']].max() -
        df[['close', 'high', 'low', 'volume', 'rsi', 'macd', 'bollinger_high', 'bollinger_low',
            'cci', 'stochastic_k', 'stochastic_d', 'atr', 'williams_r', 'momentum', 'vwap', 'ema20']].min())

    return df


# 기술적 지표 계산 함수들
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal


def compute_cci(df, period=20):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    sma = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    cci = (typical_price - sma) / (0.015 * mean_deviation)
    return cci


def compute_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k, d


def compute_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def compute_williams_r(high, low, close, period=14):
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r


def compute_momentum(series, period=10):
    return series.diff(periods=period)


def compute_vwap(df):
    q = df['volume']
    p = df['close']
    vwap = (p * q).cumsum() / q.cumsum()
    return vwap
