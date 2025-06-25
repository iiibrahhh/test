# oscillators_2.py 

import pandas as pd
import numpy as np
import time
import traceback
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import urllib.parse

# ------------------------------
# إعداد الاتصال بقاعدة البيانات
# ------------------------------
DB_CONN_STR = "Driver={ODBC Driver 17 for SQL Server};Server=localhost;Database=CryptoDB;Trusted_Connection=yes;"

def get_engine():
    conn_str_encoded = urllib.parse.quote_plus(DB_CONN_STR)
    return create_engine(f"mssql+pyodbc:///?odbc_connect={conn_str_encoded}")

engine = get_engine()

# ------------------------------
# دوال المساعدة العامة
# ------------------------------
def get_symbols():
    with engine.connect() as conn:
        df = pd.read_sql("SELECT DISTINCT symbol FROM CandleData", conn)
    return df['symbol'].dropna().tolist()

def get_candle_data(symbol, interval):
    with engine.connect() as conn:
        query = text("""
            SELECT * FROM CandleData 
            WHERE symbol = :symbol AND interval = :interval
            ORDER BY open_time ASC
        """)
        df = pd.read_sql(query, conn, params={"symbol": symbol, "interval": interval})
    return df

def is_candle_complete(interval, open_time, current_time):
    delta_map = {
        '5m': timedelta(minutes=5),
        '15m': timedelta(minutes=15),
        '30m': timedelta(minutes=30),
        '1h': timedelta(hours=1),
        '2h': timedelta(hours=2),
        '6h': timedelta(hours=6),
        '1d': timedelta(days=1)
    }
    return current_time >= open_time + delta_map[interval]

def is_candle_in_db(symbol, interval, open_time):
    table_name = f"OscillatorPerCandle_{interval}"
    with engine.connect() as conn:
        query = text(f"SELECT COUNT(*) FROM {table_name} WHERE symbol = :symbol AND open_time = :open_time")
        result = conn.execute(query, {"symbol": symbol, "open_time": open_time}).scalar()
    return result > 0










def get_intervals_to_process():
    return ['5m', '15m', '30m', '1h', '2h', '6h', '1d']


FORCE_ALL_INTERVALS = True  # ← اجعلها False للتشغيل التلقائي فقط

def get_intervals_to_process():
    if FORCE_ALL_INTERVALS:
        return ['5m', '15m', '30m', '1h', '2h', '6h', '1d']
    
    now = datetime.utcnow()
    minute = now.minute
    hour = now.hour
    intervals = []
    if minute % 5 == 0:
        intervals.append('5m')
    if minute % 15 == 0:
        intervals.append('15m')
    if minute % 30 == 0:
        intervals.append('30m')
    if minute == 0:
        intervals.append('1h')
        if hour % 2 == 0:
            intervals.append('2h')
        if hour % 6 == 0:
            intervals.append('6h')
        if hour == 0:
            intervals.append('1d')
    return intervals
















def save_to_db(symbol, interval,open_time, df):
    table_name = f"OscillatorPerCandle_{interval}"
    
    # هذه قائمة الأعمدة التي الجدول يحتويها
    allowed_columns = [
        'symbol', 'interval', 'open_time',
        'open_price', 'high_price', 'low_price', 'close_price', 'volume',
        'RSI', 'Stoch_K', 'Stoch_D', 'MACD_Line', 'MACD_Signal', 'MACD_Hist',
        'AO', 'CCI', 'MFI', 'SMA_20', 'EMA_20', 'ADX',
        'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_SpanA', 'Ichimoku_SpanB',
        'Ichimoku_Chikou', 'SAR', 'SuperTrend', 'ATR', 'NATR',
        'BB_Upper', 'BB_Lower', 'BB_Middle',
        'Donchian_Upper', 'Donchian_Lower', 'Donchian_Middle',
        'Keltner_Upper', 'Keltner_Lower', 'Keltner_Middle',
        'STC', 'Pivot', 'R1', 'R2', 'S1', 'S2',
        'Fib_0_0', 'Fib_0_236', 'Fib_0_382', 'Fib_0_5', 'Fib_0_618', 'Fib_1_0',
        'Fractal_Up', 'Fractal_Down',
        'PriceChannel_High', 'PriceChannel_Low',
        'OBV', 'Volume_ROC', 'CMF', 'VWAP',
        'Stoch_D_mask', 'AO_mask', 'CCI_mask', 'SMA_20_mask', 'ADX_mask',
        'Ichimoku_Kijun_mask', 'Ichimoku_SpanA_mask', 'Ichimoku_SpanB_mask',
        'Ichimoku_Chikou_mask', 'BB_Upper_mask', 'BB_Lower_mask', 'BB_Middle_mask',
        'Donchian_Upper_mask', 'Donchian_Lower_mask', 'Donchian_Middle_mask',
        'Keltner_Upper_mask', 'Keltner_Lower_mask',
        'Fib_0_0_mask', 'Fib_0_236_mask', 'Fib_0_382_mask', 'Fib_0_5_mask',
        'Fib_0_618_mask', 'Fib_1_0_mask',
        'Fractal_Up_mask', 'Fractal_Down_mask',
        'PriceChannel_High_mask', 'PriceChannel_Low_mask',
        'CMF_mask', 'Volume_Change_Pct'
    ]
    
    # قراءة بيانات الشموع المفتوحة المخزنة سابقًا لتجنب التكرار
    with engine.connect() as conn:
        existing = pd.read_sql(
            text(f"SELECT open_time FROM {table_name} WHERE symbol = :symbol"),
            conn,
            params={"symbol": symbol}
        )
    existing_times = set(existing['open_time'])

    rows = []
    for _, row in df.iterrows():
        assert 'open_time' in df.columns, "❌ عمود open_time غير موجود في DataFrame"

        
        row_data = row.to_dict()
        row_data['symbol'] = symbol
        row_data['interval'] = interval

        # احتفظ فقط بالأعمدة الموجودة في الجدول
        filtered_row = {k: row_data.get(k, None) for k in allowed_columns}
        rows.append(filtered_row)

    if rows:
        df_out = pd.DataFrame(rows)
        with engine.begin() as conn:
            df_out.to_sql(table_name, con=conn, if_exists='append', index=False)


# ------------------------------
# دالة تحليل المؤشرات (placeholder)
# ------------------------------

import numpy as np
import pandas as pd

def calculate_rsi_method1(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_rsi_method2(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_rsi(df, period=14):
    rsi = calculate_rsi_method1(df['close_price'], period)
    if rsi.isna().any():
        rsi = calculate_rsi_method2(df['close_price'], period)
    return rsi.fillna(method='bfill').fillna(method='ffill')

# ---------- Stochastic Oscillator (K and D) ----------
def calculate_stoch_k_method1(df, k_period=14, d_period=3):
    low_min = df['low_price'].rolling(k_period).min()
    high_max = df['high_price'].rolling(k_period).max()
    stoch_k = 100 * (df['close_price'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(d_period).mean()
    return stoch_k, stoch_d

def calculate_stoch_k_method2(df, k_period=14, d_period=3):
    # نفس الطريقة مع ewm (تجربة بديلة)
    low_min = df['low_price'].rolling(k_period).min()
    high_max = df['high_price'].rolling(k_period).max()
    stoch_k = 100 * (df['close_price'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.ewm(span=d_period, adjust=False).mean()
    return stoch_k, stoch_d

def calculate_stoch(df):
    stoch_k, stoch_d = calculate_stoch_k_method1(df)
    if stoch_k.isna().any() or stoch_d.isna().any():
        stoch_k, stoch_d = calculate_stoch_k_method2(df)
    stoch_k = stoch_k.bfill().ffill()
    stoch_d = stoch_d.bfill().ffill()
    return stoch_k, stoch_d


# ---------- MACD ----------
def calculate_macd_method1(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def calculate_macd_method2(close, fast=12, slow=26, signal=9):
    # بديل مشابه مع adjust=True (أحيانًا يعطي نتائج مختلفة قليلاً)
    ema_fast = close.ewm(span=fast, adjust=True).mean()
    ema_slow = close.ewm(span=slow, adjust=True).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=True).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def calculate_macd(df):
    macd_line, macd_signal, macd_hist = calculate_macd_method1(df['close_price'])
    if macd_line.isna().any() or macd_signal.isna().any() or macd_hist.isna().any():
        macd_line, macd_signal, macd_hist = calculate_macd_method2(df['close_price'])
    macd_line = macd_line.fillna(method='bfill').fillna(method='ffill')
    macd_signal = macd_signal.fillna(method='bfill').fillna(method='ffill')
    macd_hist = macd_hist.fillna(method='bfill').fillna(method='ffill')
    return macd_line, macd_signal, macd_hist

# ---------- Awesome Oscillator ----------
def calculate_ao_method1(df):
    median_price = (df['high_price'] + df['low_price']) / 2
    ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
    return ao

def calculate_ao_method2(df):
    median_price = (df['high_price'] + df['low_price']) / 2
    ao = median_price.ewm(span=5, adjust=False).mean() - median_price.ewm(span=34, adjust=False).mean()
    return ao

def calculate_ao(df):
    ao = calculate_ao_method1(df)
    if ao.isna().any():
        ao = calculate_ao_method2(df)
    return ao.fillna(method='bfill').fillna(method='ffill')

# ---------- CCI ----------
def calculate_cci_method1(df, period=20):
    tp = (df['high_price'] + df['low_price'] + df['close_price']) / 3
    ma = tp.rolling(window=period).mean()
    md = tp.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean())
    cci = (tp - ma) / (0.015 * md)
    return cci

def calculate_cci_method2(df, period=20):
    tp = (df['high_price'] + df['low_price'] + df['close_price']) / 3
    ma = tp.ewm(span=period, adjust=False).mean()
    md = tp.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean())
    cci = (tp - ma) / (0.015 * md)
    return cci

def calculate_cci(df):
    cci = calculate_cci_method1(df)
    if cci.isna().any():
        cci = calculate_cci_method2(df)
    return cci.fillna(method='bfill').fillna(method='ffill')

# ---------- MFI ----------
def calculate_mfi_method1(df, period=14):
    typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = []
    negative_flow = []

    for i in range(1, len(df)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.append(money_flow.iloc[i])
            negative_flow.append(0)
        else:
            positive_flow.append(0)
            negative_flow.append(money_flow.iloc[i])

    positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=period).sum()

    mfi = 100 * positive_mf / (positive_mf + negative_mf)
    return mfi

def calculate_mfi_method2(df, period=14):
    # بديل ewm مع money flow
    typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
    money_flow = typical_price * df['volume']
    delta = typical_price.diff()
    positive_flow = money_flow.where(delta > 0, 0)
    negative_flow = money_flow.where(delta < 0, 0).abs()

    positive_mf = positive_flow.ewm(span=period, adjust=False).mean()
    negative_mf = negative_flow.ewm(span=period, adjust=False).mean()

    mfi = 100 * positive_mf / (positive_mf + negative_mf)
    return mfi

def calculate_mfi(df):
    mfi = calculate_mfi_method1(df)
    if mfi.isna().any():
        mfi = calculate_mfi_method2(df)
    return mfi.fillna(method='bfill').fillna(method='ffill')

# ---------- STC (Schaff Trend Cycle) ----------
def calculate_stc(df, short_cycle=23, long_cycle=50, cycle=10):
    # STC معتمدة على MACD مع إضافات
    macd_line, macd_signal, _ = calculate_macd(df)
    macd_diff = macd_line - macd_signal
    stc_fast = macd_diff.ewm(span=short_cycle, adjust=False).mean()
    stc_slow = stc_fast.ewm(span=long_cycle, adjust=False).mean()
    stc = 100 * (stc_fast - stc_slow) / (stc_fast.max() - stc_slow.min() + 1e-10)
    return stc.fillna(method='bfill').fillna(method='ffill')

# ---------- Moving Averages ----------
def calculate_sma(df, period=20):
    return df['close_price'].rolling(window=period).mean().fillna(method='bfill').fillna(method='ffill')

def calculate_ema(df, period=20):
    return df['close_price'].ewm(span=period, adjust=False).mean().fillna(method='bfill').fillna(method='ffill')

# ---------- ADX ----------
def calculate_adx(df, period=14):
    high = df['high_price']
    low = df['low_price']
    close = df['close_price']

    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()
    return adx.fillna(method='bfill').fillna(method='ffill')

# ---------- Ichimoku ----------
def calculate_ichimoku(df):
    high_9 = df['high_price'].rolling(window=9).max()
    low_9 = df['low_price'].rolling(window=9).min()
    tenkan = (high_9 + low_9) / 2

    high_26 = df['high_price'].rolling(window=26).max()
    low_26 = df['low_price'].rolling(window=26).min()
    kijun = (high_26 + low_26) / 2

    span_a = ((tenkan + kijun) / 2).shift(26)

    high_52 = df['high_price'].rolling(window=52).max()
    low_52 = df['low_price'].rolling(window=52).min()
    span_b = ((high_52 + low_52) / 2).shift(52)

    chikou = df['close_price'].shift(-26)
    return tenkan.fillna(method='bfill'), kijun.fillna(method='bfill'), span_a.fillna(method='bfill'), span_b.fillna(method='bfill'), chikou.fillna(method='bfill')

# ---------- SAR ----------
def calculate_sar(df, af_start=0.02, af_step=0.02, af_max=0.2):
    high = df['high_price']
    low = df['low_price']
    length = len(df)
    sar = pd.Series(np.nan, index=df.index)
    trend = True  # True=uptrend, False=downtrend
    af = af_start
    ep = low.iloc[0]
    sar.iloc[0] = low.iloc[0]

    for i in range(1, length):
        prev_sar = sar.iloc[i - 1]
        if trend:
            sar.iloc[i] = prev_sar + af * (ep - prev_sar)
            if high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + af_step, af_max)
            if low.iloc[i] < sar.iloc[i]:
                trend = False
                sar.iloc[i] = ep
                ep = low.iloc[i]
                af = af_start
        else:
            sar.iloc[i] = prev_sar - af * (prev_sar - ep)
            if low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + af_step, af_max)
            if high.iloc[i] > sar.iloc[i]:
                trend = True
                sar.iloc[i] = ep
                ep = high.iloc[i]
                af = af_start
    return sar.fillna(method='bfill').fillna(method='ffill')

# ---------- SuperTrend ----------
def calculate_supertrend(df, period=10, multiplier=3):
    hl2 = (df['high_price'] + df['low_price']) / 2
    atr = calculate_atr(df, period)
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    supertrend = pd.Series(True, index=df.index)  # True=uptrend, False=downtrend
    for i in range(1, len(df)):
        if df['close_price'].iloc[i] > upperband.iloc[i-1]:
            supertrend.iloc[i] = True
        elif df['close_price'].iloc[i] < lowerband.iloc[i-1]:
            supertrend.iloc[i] = False
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
            if supertrend.iloc[i] and lowerband.iloc[i] < lowerband.iloc[i-1]:
                lowerband.iloc[i] = lowerband.iloc[i-1]
            if not supertrend.iloc[i] and upperband.iloc[i] > upperband.iloc[i-1]:
                upperband.iloc[i] = upperband.iloc[i-1]
    return supertrend.astype(int)

# ---------- ATR ----------
def calculate_atr(df, period=14):
    high = df['high_price']
    low = df['low_price']
    close = df['close_price']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.fillna(method='bfill').fillna(method='ffill')

# ---------- NATR ----------
def calculate_natr(df, period=14):
    atr = calculate_atr(df, period)
    natr = 100 * atr / df['close_price']
    return natr.fillna(method='bfill').fillna(method='ffill')

# ---------- Bollinger Bands ----------
def calculate_bollinger_bands(df, period=20, std_dev=2):
    sma = df['close_price'].rolling(window=period).mean()
    std = df['close_price'].rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    middle = sma
    return upper.fillna(method='bfill').fillna(method='ffill'), lower.fillna(method='bfill').fillna(method='ffill'), middle.fillna(method='bfill').fillna(method='ffill')

# ---------- Donchian Channels ----------
def calculate_donchian_channels(df, period=20):
    upper = df['high_price'].rolling(window=period).max()
    lower = df['low_price'].rolling(window=period).min()
    middle = (upper + lower) / 2
    return upper.fillna(method='bfill').fillna(method='ffill'), lower.fillna(method='bfill').fillna(method='ffill'), middle.fillna(method='bfill').fillna(method='ffill')

# ---------- Keltner Channels ----------
def calculate_keltner_channels(df, period=20, multiplier=1.5):
    ema = df['close_price'].ewm(span=period, adjust=False).mean()
    atr = calculate_atr(df, period)
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    middle = ema
    return upper.fillna(method='bfill').fillna(method='ffill'), lower.fillna(method='bfill').fillna(method='ffill'), middle.fillna(method='bfill').fillna(method='ffill')

# ---------- Pivot Points ----------
def calculate_pivot_points(df):
    high = df['high_price'].shift(1)
    low = df['low_price'].shift(1)
    close = df['close_price'].shift(1)
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    return pivot.fillna(method='bfill').fillna(method='ffill'), r1.fillna(method='bfill').fillna(method='ffill'), r2.fillna(method='bfill').fillna(method='ffill'), s1.fillna(method='bfill').fillna(method='ffill'), s2.fillna(method='bfill').fillna(method='ffill')

# ---------- Fibonacci Levels ----------
def calculate_fibonacci_levels(df):
    low = df['low_price'].rolling(window=20).min()
    high = df['high_price'].rolling(window=20).max()
    diff = high - low
    fib_0_0 = low
    fib_0_236 = low + 0.236 * diff
    fib_0_382 = low + 0.382 * diff
    fib_0_5 = low + 0.5 * diff
    fib_0_618 = low + 0.618 * diff
    fib_1_0 = high
    return fib_0_0.fillna(method='bfill').fillna(method='ffill'), fib_0_236.fillna(method='bfill').fillna(method='ffill'), fib_0_382.fillna(method='bfill').fillna(method='ffill'), fib_0_5.fillna(method='bfill').fillna(method='ffill'), fib_0_618.fillna(method='bfill').fillna(method='ffill'), fib_1_0.fillna(method='bfill').fillna(method='ffill')

# ---------- Fractal ----------
def calculate_fractal(df):
    high = df['high_price']
    low = df['low_price']
    fractal_up = (high.shift(2) < high.shift(1)) & (high.shift(1) < high) & (high.shift(1) > high.shift(-1)) & (high.shift(-1) > high.shift(-2))
    fractal_down = (low.shift(2) > low.shift(1)) & (low.shift(1) > low) & (low.shift(1) < low.shift(-1)) & (low.shift(-1) < low.shift(-2))
    fractal_up = fractal_up.astype(int)
    fractal_down = fractal_down.astype(int)
    return fractal_up.fillna(0), fractal_down.fillna(0)

# ---------- Price Channel ----------
def calculate_price_channel(df, period=20):
    high = df['high_price'].rolling(window=period).max()
    low = df['low_price'].rolling(window=period).min()
    return high.fillna(method='bfill').fillna(method='ffill'), low.fillna(method='bfill').fillna(method='ffill')

# ---------- Volume Indicators ----------
def calculate_obv(df):
    obv = (np.sign(df['close_price'].diff()) * df['volume']).fillna(0).cumsum()
    return obv.fillna(method='bfill').fillna(method='ffill')

def calculate_volume_roc(df, period=10):
    vol = df['volume']
    roc = vol.pct_change(periods=period) * 100
    return roc.fillna(method='bfill').fillna(method='ffill')

def calculate_cmf(df, period=20):
    mfv = ((df['close_price'] - df['low_price']) - (df['high_price'] - df['close_price'])) / (df['high_price'] - df['low_price'])
    mfv = mfv.fillna(0)
    mfv_volume = mfv * df['volume']
    cmf = mfv_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    return cmf.fillna(method='bfill').fillna(method='ffill')

def calculate_vwap(df):
    cum_vol_price = (df['close_price'] * df['volume']).cumsum()
    cum_volume = df['volume'].cumsum()
    vwap = cum_vol_price / cum_volume
    return vwap.fillna(method='bfill').fillna(method='ffill')

# ---------- Volume Change Percent ----------
def calculate_volume_change_pct(df):
    volume_pct = df['volume'].pct_change() * 100
    return volume_pct.fillna(0)

# ---------- دالة رئيسية لحساب جميع المؤشرات وإرجاع DataFrame ----------
def calculate_all_indicators(df):
    df_ind = pd.DataFrame(index=df.index)

    # مؤشر RSI
    df_ind['RSI'] = calculate_rsi(df)

    # Stochastic
    df_ind['Stoch_K'], df_ind['Stoch_D'] = calculate_stoch(df)

    # MACD
    df_ind['MACD_Line'], df_ind['MACD_Signal'], df_ind['MACD_Hist'] = calculate_macd(df)

    # AO
    df_ind['AO'] = calculate_ao(df)

    # CCI
    df_ind['CCI'] = calculate_cci(df)

    # MFI
    df_ind['MFI'] = calculate_mfi(df)

    # STC
    df_ind['STC'] = calculate_stc(df)

    # SMA, EMA
    df_ind['SMA_20'] = calculate_sma(df)
    df_ind['EMA_20'] = calculate_ema(df)

    # ADX
    df_ind['ADX'] = calculate_adx(df)

    # Ichimoku
    df_ind['Ichimoku_Tenkan'], df_ind['Ichimoku_Kijun'], df_ind['Ichimoku_SpanA'], df_ind['Ichimoku_SpanB'], df_ind['Ichimoku_Chikou'] = calculate_ichimoku(df)

    # SAR
    df_ind['SAR'] = calculate_sar(df)

    # SuperTrend
    df_ind['SuperTrend'] = calculate_supertrend(df)

    # ATR, NATR
    df_ind['ATR'] = calculate_atr(df)
    df_ind['NATR'] = calculate_natr(df)

    # Bollinger Bands
    df_ind['BB_Upper'], df_ind['BB_Lower'], df_ind['BB_Middle'] = calculate_bollinger_bands(df)

    # Donchian Channels
    df_ind['Donchian_Upper'], df_ind['Donchian_Lower'], df_ind['Donchian_Middle'] = calculate_donchian_channels(df)

    # Keltner Channels
    df_ind['Keltner_Upper'], df_ind['Keltner_Lower'], df_ind['Keltner_Middle'] = calculate_keltner_channels(df)

    # Pivot Points
    df_ind['Pivot'], df_ind['R1'], df_ind['R2'], df_ind['S1'], df_ind['S2'] = calculate_pivot_points(df)

    # Fibonacci Levels
    df_ind['Fib_0_0'], df_ind['Fib_0_236'], df_ind['Fib_0_382'], df_ind['Fib_0_5'], df_ind['Fib_0_618'], df_ind['Fib_1_0'] = calculate_fibonacci_levels(df)

    # Fractal
    df_ind['Fractal_Up'], df_ind['Fractal_Down'] = calculate_fractal(df)

    # Price Channel
    df_ind['PriceChannel_High'], df_ind['PriceChannel_Low'] = calculate_price_channel(df)

    # Volume Indicators
    df_ind['OBV'] = calculate_obv(df)
    df_ind['Volume_ROC'] = calculate_volume_roc(df)
    df_ind['CMF'] = calculate_cmf(df)
    df_ind['VWAP'] = calculate_vwap(df)

    # Volume Change Percent
    df_ind['Volume_Change_Pct'] = calculate_volume_change_pct(df)

    return df_ind




#                                mask




def add_raw_columns_and_masks(df, df_indicators):
    # نسخ الأعمدة الخام
    df_indicators['open_price'] = df['open_price']
    df_indicators['high_price'] = df['high_price']
    df_indicators['low_price'] = df['low_price']
    df_indicators['close_price'] = df['close_price']
    df_indicators['volume'] = df['volume']
    
    # أعمدة الماسكات (مثلاً نعتبرها أعمدة منطقية تدل على وجود قيم غير صالحة أو نان في الأعمدة المذكورة)
    masks = [
        'Stoch_D_mask',
        'AO_mask',
        'CCI_mask',
        'SMA_20_mask',
        'ADX_mask',
        'Ichimoku_Kijun_mask',
        'Ichimoku_SpanA_mask',
        'Ichimoku_SpanB_mask',
        'Ichimoku_Chikou_mask',
        'BB_Upper_mask',
        'BB_Lower_mask',
        'BB_Middle_mask',
        'Donchian_Upper_mask',
        'Donchian_Lower_mask',
        'Donchian_Middle_mask',
        'Keltner_Upper_mask',
        'Keltner_Lower_mask',
        'Fib_0_0_mask',
        'Fib_0_236_mask',
        'Fib_0_382_mask',
        'Fib_0_5_mask',
        'Fib_0_618_mask',
        'Fib_1_0_mask',
        'Fractal_Up_mask',
        'Fractal_Down_mask',
        'PriceChannel_High_mask',
        'PriceChannel_Low_mask',
        'CMF_mask',
    ]
    
    # إضافة الماسكات: 
    # لو هي موجودة مسبقاً في df أو df_indicators تنسخ، وإلا تنشئها مثلاً كـ isna() للأعمدة المقابلة
    # مثال عام - تعديل حسب المكان الفعلي للبيانات:
    for mask_col in masks:
        if mask_col in df_indicators.columns:
            continue  # موجودة أصلاً
        # مثلا نحسبها حسب وجود NaN في العمود بدون "_mask" المكافئ:
        base_col = mask_col.replace('_mask', '')
        if base_col in df_indicators.columns:
            df_indicators[mask_col] = df_indicators[base_col].isna().astype(int)
        else:
            df_indicators[mask_col] = 0  # أو NaN حسب الحاجة
    
    # حساب نسبة تغير الحجم (Volume_Change_Pct)
    df_indicators['Volume_Change_Pct'] = df['volume'].pct_change().fillna(0) * 100
    
    return df_indicators











def main():
    print("\U0001F680 بدء تشغيل تحليل المؤشرات لكل فريم")
    pending_candles = {}

    while True:
        now = datetime.utcnow()
        intervals = get_intervals_to_process()
        if not intervals:
            time.sleep(60)
            continue

        symbols = get_symbols()

        # معالجة الشموع المتأخرة
        for (symbol, tf), open_time in list(pending_candles.items()):
            df = get_candle_data(symbol, tf)
            if df.empty:
                continue
            last_time = df['open_time'].max()
            if is_candle_complete(tf, last_time, now):
                df_ind = calculate_all_indicators(df)
                df_ind['symbol'] = symbol
                df_ind['interval'] = tf
                df_ind['open_time'] = df['open_time']
                df_ind = add_raw_columns_and_masks(df, df_ind)
                save_to_db(symbol, tf, df['open_time'], df_ind)

                del pending_candles[(symbol, tf)]
                print("🚀 بدء الحفظ")

        # تحليل الفريمات الجارية
        for symbol in symbols:
            for tf in intervals:
                try:
                    df = get_candle_data(symbol, tf)
                    if df.empty:
                        continue
                    last_open = df['open_time'].max()
                    if not is_candle_complete(tf, last_open, now):
                        pending_candles[(symbol, tf)] = last_open
                        continue
                    if is_candle_in_db(symbol, tf, last_open):
                        continue
                    df_ind = calculate_all_indicators(df)
                    df_ind['symbol'] = symbol
                    df_ind['interval'] = tf
                    df_ind['open_time'] = df['open_time']
                    df_ind = add_raw_columns_and_masks(df, df_ind)
                    save_to_db(symbol, tf, df['open_time'], df_ind)

                except Exception as e:
                    print(f"❌ خطأ في {symbol} - {tf}: {e}")
                    traceback.print_exc()
                time.sleep(60)

if __name__ == "__main__":
    main()
