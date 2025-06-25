# oscillators_2.py - النسخة المحسنة

import pandas as pd
import numpy as np
import traceback
from datetime import datetime
from sqlalchemy import create_engine, text
import urllib.parse

FUTURE_OFFSET = 2  # عدد الشموع المستقبلية المطلوبة لتأخير المؤشرات
FRACTAL_OFFSET = 2

# تعريف offsets لكل مؤشر حسب الحاجة
FUTURE_OFFSET_ICHIMOKU = 26  # Ichimoku يحتاج 26 شمعة للأمام
FUTURE_OFFSET_OTHERS = 2     # مؤشرات أخرى تحتاج 2 شمعة فقط (أو أي قيمة مناسبة)

# ------------------------------
# إعداد الاتصال بقاعدة البيانات
# ------------------------------
DB_CONN_STR = "Driver={ODBC Driver 17 for SQL Server};Server=localhost;Database=CryptoDB;Trusted_Connection=yes;"

def get_engine():
    conn_str_encoded = urllib.parse.quote_plus(DB_CONN_STR)
    return create_engine(f"mssql+pyodbc:///?odbc_connect={conn_str_encoded}")

engine = get_engine()


# ------------------------------
# دوال المساعدة
# ------------------------------
def get_symbols():
    with engine.connect() as conn:
        df = pd.read_sql("SELECT DISTINCT symbol FROM CandleData WHERE interval = '5m'", conn)
    return df['symbol'].dropna().tolist()

def get_candle_data(symbol):
    with engine.connect() as conn:
        query = text("""
            SELECT * FROM CandleData 
            WHERE symbol = :symbol AND interval = '5m'
            ORDER BY open_time ASC
        """)
        df = pd.read_sql(query, conn, params={"symbol": symbol})
    return df


def load_and_validate_data(symbol):
    df = get_candle_data(symbol)
    if df.empty or len(df) < 60:
        return None

    df = df.tail(900)  # نحتفظ بآخر 600 شمعة

    last_saved_time = get_latest_saved_open_time(symbol)
    if last_saved_time:
        df = df[df['open_time'] > last_saved_time]

    if df.empty or len(df) < 10:
        return None

    return df.reset_index(drop=True)



def get_latest_saved_open_time(symbol):
    table_name = "OscillatorPerCandle_5m"
    with engine.connect() as conn:
        query = text(f"""
            SELECT MAX(open_time) as last_time FROM {table_name}
            WHERE symbol = :symbol
        """)
        result = conn.execute(query, {"symbol": symbol}).fetchone()
    return result.last_time if result and result.last_time else None









def save_to_db(symbol, df):
    table_name = "OscillatorPerCandle_5m"
    if df.empty:
        return

    df['symbol'] = symbol
    df['interval'] = '5m'

    # **تأكد من إضافة كل الأعمدة المطلوبة، خاصة أعمدة الماسك**
    allowed_columns = {
        'symbol', 'interval', 'open_time', 'RSI', 'RSI_mask', 'SuperTrend', 'SuperTrend_mask',
        # أضف بقية الأعمدة والمؤشرات وأعمدة الماسك هنا بالكامل
        'Stoch_K', 'Stoch_D', 'Stoch_K_mask', 'Stoch_D_mask',
        'MACD_Line', 'MACD_Signal', 'MACD_Hist', 'MACD_Line_mask', 'MACD_Signal_mask', 'MACD_Hist_mask',
        'AO', 'AO_mask', 'CCI', 'CCI_mask', 'MFI', 'MFI_mask', 'SMA_20', 'SMA_20_mask', 'EMA_20', 'EMA_20_mask',
        'ADX', 'ADX_mask', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_SpanA', 'Ichimoku_SpanB', 'Ichimoku_Chikou',
        'Ichimoku_Kijun_mask', 'Ichimoku_SpanA_mask', 'Ichimoku_SpanB_mask', 'Ichimoku_Chikou_mask',
        'SAR', 'SAR_mask', 'ATR', 'ATR_mask', 'NATR', 'NATR_mask',
        'BB_Upper', 'BB_Lower', 'BB_Middle', 'BB_Upper_mask', 'BB_Lower_mask', 'BB_Middle_mask',
        'Donchian_Upper', 'Donchian_Lower', 'Donchian_Middle',
        'Donchian_Upper_mask', 'Donchian_Lower_mask', 'Donchian_Middle_mask',
        'Keltner_Upper', 'Keltner_Lower', 'Keltner_Middle',
        'Keltner_Upper_mask', 'Keltner_Lower_mask', 'Keltner_Middle_mask',
        'STC', 'STC_mask',
        'Pivot', 'Pivot_mask', 'R1', 'R1_mask', 'R2', 'R2_mask', 'S1', 'S1_mask', 'S2', 'S2_mask',
        'Fib_0_0', 'Fib_0_236', 'Fib_0_382', 'Fib_0_5', 'Fib_0_618', 'Fib_1_0',
        'Fib_0_0_mask', 'Fib_0_236_mask', 'Fib_0_382_mask', 'Fib_0_5_mask', 'Fib_0_618_mask', 'Fib_1_0_mask',
        'Fractal_Up', 'Fractal_Down', 'Fractal_Up_mask', 'Fractal_Down_mask',
        'PriceChannel_High', 'PriceChannel_Low',
        'PriceChannel_High_mask', 'PriceChannel_Low_mask',
        'OBV', 'OBV_mask', 'Volume_ROC', 'Volume_ROC_mask', 'CMF', 'CMF_mask',
        'VWAP',  # إذا موجود
        'open_price', 'high_price', 'low_price', 'close_price', 'volume',
        'Volume_Change_Pct'
    }

    # فلترة الأعمدة حتى لا يدخل شيء غير موجود في allowed_columns
    df_to_save = df[[col for col in df.columns if col in allowed_columns]]

    with engine.begin() as conn:
        df_to_save.to_sql(table_name, con=conn, if_exists='append', index=False)


# ==========================
# RSI Calculation
# ==========================
def calculate_rsi_method1(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_rsi_method2(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_rsi(df, period=14):
    rsi = calculate_rsi_method1(df['close_price'], period)
    if rsi.isna().sum() > period:
        rsi = calculate_rsi_method2(df['close_price'], period)
    return rsi













# -------------------------
# Stochastic Oscillator (%K, %D)
# -------------------------
def calculate_stoch(df, k_period=14, d_period=3):
    low_min = df['low_price'].rolling(window=k_period).min()
    high_max = df['high_price'].rolling(window=k_period).max()
    stoch_k = 100 * (df['close_price'] - low_min) / (high_max - low_min)
    stoch_k[:k_period] = np.nan
    stoch_d = stoch_k.rolling(window=d_period).mean()
    stoch_d[:k_period + d_period - 1] = np.nan
    return stoch_k, stoch_d

# -------------------------
# MACD (Line, Signal, Histogram)
# -------------------------
def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['close_price'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close_price'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    # تعيين NaN لأول شموع غير مكتملة
    macd_line[:slow] = np.nan
    macd_signal[:slow + signal -1] = np.nan
    macd_hist[:slow + signal -1] = np.nan

    return macd_line, macd_signal, macd_hist

# -------------------------
# Awesome Oscillator (AO)
# -------------------------
def calculate_ao(df, short=5, long=34):
    median_price = (df['high_price'] + df['low_price']) / 2
    ao = median_price.rolling(window=short).mean() - median_price.rolling(window=long).mean()
    ao[:long] = np.nan
    return ao

# -------------------------
# Commodity Channel Index (CCI)
# -------------------------
def calculate_cci(df, period=20):
    tp = (df['high_price'] + df['low_price'] + df['close_price']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mean_dev = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    cci[:period] = np.nan
    return cci

# -------------------------
# Money Flow Index (MFI)
# -------------------------
def calculate_mfi(df, period=14):
    typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = []
    negative_flow = []
    for i in range(1, len(df)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.append(money_flow.iloc[i])
            negative_flow.append(0)
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            positive_flow.append(0)
            negative_flow.append(money_flow.iloc[i])
        else:
            positive_flow.append(0)
            negative_flow.append(0)
    positive_flow = pd.Series(positive_flow, index=df.index[1:])
    negative_flow = pd.Series(negative_flow, index=df.index[1:])
    # حساب النسب المتحركة
    pos_mf = positive_flow.rolling(window=period).sum()
    neg_mf = negative_flow.rolling(window=period).sum()
    mfi = 100 * pos_mf / (pos_mf + neg_mf)
    mfi = mfi.reindex(df.index)
    mfi[:period] = np.nan
    return mfi

# -------------------------
# SMA 20
# -------------------------
def calculate_sma(df, period=20):
    sma = df['close_price'].rolling(window=period).mean()
    sma[:period] = np.nan
    return sma

# -------------------------
# EMA 20
# -------------------------
def calculate_ema(df, period=20):
    ema = df['close_price'].ewm(span=period, adjust=False).mean()
    ema[:period] = np.nan
    return ema

# -------------------------
# ADX (Average Directional Index)
# -------------------------
def calculate_adx(df, period=14):
    high = df['high_price']
    low = df['low_price']
    close = df['close_price']

    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()
    adx[:period * 2] = np.nan

    return adx



def calculate_ichimoku(df):
    high_9 = df['high_price'].rolling(window=9).max()
    low_9 = df['low_price'].rolling(window=9).min()
    tenkan = (high_9 + low_9) / 2

    high_26 = df['high_price'].rolling(window=26).max()
    low_26 = df['low_price'].rolling(window=26).min()
    kijun = (high_26 + low_26) / 2

    # احذف shift من هنا
    span_a = (tenkan + kijun) / 2

    high_52 = df['high_price'].rolling(window=52).max()
    low_52 = df['low_price'].rolling(window=52).min()
    # احذف shift من هنا
    span_b = (high_52 + low_52) / 2

    chikou = df['close_price'].shift(-26)  # الإغلاق متأخر 26 شمعة للأمام (شيكو)

    # تعيين NaN للشموع الأولى حسب أطول فترة 52 (بدون 26 لأننا حذفنا shift)
    tenkan[:9] = np.nan
    kijun[:26] = np.nan
    span_a[:52] = np.nan
    span_b[:52] = np.nan
    chikou[-26:] = np.nan  # نهاية السلسلة لأن الشيكو متقدم للأمام

    return tenkan, kijun, span_a, span_b, chikou




# -------------------------
# ATR - Average True Range
# -------------------------
def calculate_atr(df, period=14):
    high_low = df['high_price'] - df['low_price']
    high_close = (df['high_price'] - df['close_price'].shift()).abs()
    low_close = (df['low_price'] - df['close_price'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    atr[:period] = np.nan
    return atr

# -------------------------
# NATR - Normalized ATR (ATR / Close * 100)
# -------------------------
def calculate_natr(df, period=14):
    atr = calculate_atr(df, period)
    natr = (atr / df['close_price']) * 100
    natr[:period] = np.nan
    return natr

# -------------------------
# SAR - Parabolic SAR (simplified implementation)
# -------------------------
def calculate_sar(df, af=0.02, max_af=0.2):
    high = df['high_price']
    low = df['low_price']
    close = df['close_price']
    sar = pd.Series(np.nan, index=df.index)

    length = len(df)
    if length < 2:
        return sar

    trend = True  # True = uptrend, False = downtrend
    af_current = af
    ep = low[0]  # extreme point
    sar.iloc[0] = low[0]

    for i in range(1, length):
        prev_sar = sar.iloc[i-1]
        if trend:
            sar.iloc[i] = prev_sar + af_current * (ep - prev_sar)
            if low[i] < sar.iloc[i]:
                trend = False
                sar.iloc[i] = ep
                ep = low[i]
                af_current = af
            else:
                if high[i] > ep:
                    ep = high[i]
                    af_current = min(af_current + af, max_af)
        else:
            sar.iloc[i] = prev_sar - af_current * (prev_sar - ep)
            if high[i] > sar.iloc[i]:
                trend = True
                sar.iloc[i] = ep
                ep = high[i]
                af_current = af
            else:
                if low[i] < ep:
                    ep = low[i]
                    af_current = min(af_current + af, max_af)
    return sar

# -------------------------
# SuperTrend (using ATR)
# -------------------------
def calculate_supertrend(df, period=10, multiplier=3):
    hl2 = (df['high_price'] + df['low_price']) / 2
    atr = calculate_atr(df, period)
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    supertrend = pd.Series(np.nan, index=df.index)
    trend = True  # True = uptrend, False = downtrend
    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = True
            continue

        if df['close_price'].iloc[i] > upperband.iloc[i-1]:
            trend = True
        elif df['close_price'].iloc[i] < lowerband.iloc[i-1]:
            trend = False

        supertrend.iloc[i] = trend

        # تعديل upperband و lowerband لمنع التذبذب (يمكن إضافة تحسينات)
    return supertrend.astype(int)  # 1 = uptrend, 0 = downtrend

# -------------------------
# Bollinger Bands (Upper, Lower, Middle)
# -------------------------
def calculate_bollinger_bands(df, period=20, std_dev=2):
    middle_band = df['close_price'].rolling(window=period).mean()
    std = df['close_price'].rolling(window=period).std()
    upper_band = middle_band + std_dev * std
    lower_band = middle_band - std_dev * std

    upper_band[:period] = np.nan
    middle_band[:period] = np.nan
    lower_band[:period] = np.nan

    return upper_band, lower_band, middle_band

# -------------------------
# Donchian Channels (Upper, Lower, Middle)
# -------------------------
def calculate_donchian(df, period=20):
    upper = df['high_price'].rolling(window=period).max()
    lower = df['low_price'].rolling(window=period).min()
    middle = (upper + lower) / 2

    upper[:period] = np.nan
    lower[:period] = np.nan
    middle[:period] = np.nan

    return upper, lower, middle


# -------------------------
# Donchian Channels (Upper, Lower, Middle)
# -------------------------
def calculate_donchian(df, period=20):
    donchian_upper = df['high_price'].rolling(window=period).max()
    donchian_lower = df['low_price'].rolling(window=period).min()
    donchian_middle = (donchian_upper + donchian_lower) / 2

    donchian_upper[:period] = np.nan
    donchian_lower[:period] = np.nan
    donchian_middle[:period] = np.nan

    return donchian_upper, donchian_lower, donchian_middle

# -------------------------
# Keltner Channels (Upper, Lower, Middle)
# -------------------------
def calculate_keltner(df, period=20, atr_period=10, multiplier=1.5):
    ema = df['close_price'].ewm(span=period, adjust=False).mean()
    atr = calculate_atr(df, period=atr_period)

    keltner_upper = ema + multiplier * atr
    keltner_lower = ema - multiplier * atr
    keltner_middle = ema

    keltner_upper[:max(period, atr_period)] = np.nan
    keltner_lower[:max(period, atr_period)] = np.nan
    keltner_middle[:max(period, atr_period)] = np.nan

    return keltner_upper, keltner_lower, keltner_middle



def calculate_stc(df, short_cycle=23, long_cycle=50, cycle=10):
    # STC يحتاج EMA و MACD أساسيين.
    # هذه نسخة مبسطة، تعتمد على MACD وتغييراته.
    macd_line, macd_signal, _ = calculate_macd(df, fast=short_cycle, slow=long_cycle, signal=cycle)
    macd_diff = macd_line - macd_signal

    # حساب %K و %D مثل Stochastic على macd_diff
    min_val = macd_diff.rolling(window=cycle).min()
    max_val = macd_diff.rolling(window=cycle).max()
    stc = 100 * (macd_diff - min_val) / (max_val - min_val)

    stc[:(long_cycle + cycle)] = np.nan
    return stc

def calculate_pivot_points(df):
    # عادة تحسب لنقاط اليوم التالي بناءً على بيانات اليوم السابق.
    # هنا نستخدم الشمعة السابقة فقط (يمكن تعديلها حسب الحاجة)

    pivot = (df['high_price'].shift(1) + df['low_price'].shift(1) + df['close_price'].shift(1)) / 3
    r1 = 2 * pivot - df['low_price'].shift(1)
    s1 = 2 * pivot - df['high_price'].shift(1)
    r2 = pivot + (r1 - s1)
    s2 = pivot - (r1 - s1)

    pivot[:1] = np.nan
    r1[:1] = np.nan
    s1[:1] = np.nan
    r2[:1] = np.nan
    s2[:1] = np.nan

    return pivot, r1, r2, s1, s2


def calculate_fibonacci_levels(df, period=20):
    high_max = df['high_price'].rolling(window=period).max()
    low_min = df['low_price'].rolling(window=period).min()

    fib_0_0 = low_min
    fib_0_236 = low_min + 0.236 * (high_max - low_min)
    fib_0_382 = low_min + 0.382 * (high_max - low_min)
    fib_0_5 = low_min + 0.5 * (high_max - low_min)
    fib_0_618 = low_min + 0.618 * (high_max - low_min)

    fib_0_0[:period] = np.nan
    fib_0_236[:period] = np.nan
    fib_0_382[:period] = np.nan
    fib_0_5[:period] = np.nan
    fib_0_618[:period] = np.nan

    return fib_0_0, fib_0_236, fib_0_382, fib_0_5, fib_0_618



def calculate_fib_1_0(df, period=20):
    high_max = df['high_price'].rolling(window=period).max()
    fib_1_0 = high_max
    fib_1_0[:period] = np.nan
    return fib_1_0



def calculate_fractal_up(df):
    high = df['high_price']
    # نحلل فقط حتى الشمعة رقم len(df) - 2، ونترك آخر شمعتين NaN
    fractal_up = (
        (high.shift(2) < high) &
        (high.shift(1) < high) &
        (high.shift(-1) < high) &
        (high.shift(-2) < high)
    )
    fractal_up = fractal_up.astype(int).replace(0, np.nan)

    # تعيين NaN للآخرين لأنهم لا يمكن حسابهم
    fractal_up.iloc[-2:] = np.nan

    return fractal_up

def calculate_fractal_down(df):
    low = df['low_price']
    fractal_down = (
        (low.shift(2) > low) &
        (low.shift(1) > low) &
        (low.shift(-1) > low) &
        (low.shift(-2) > low)
    )
    fractal_down = fractal_down.astype(int).replace(0, np.nan)

    fractal_down.iloc[-2:] = np.nan

    return fractal_down


def calculate_price_channel_high(df, period=20):
    price_channel_high = df['high_price'].rolling(window=period).max()
    price_channel_high[:period] = np.nan
    return price_channel_high

def calculate_price_channel_low(df, period=20):
    price_channel_low = df['low_price'].rolling(window=period).min()
    price_channel_low[:period] = np.nan
    return price_channel_low



def calculate_obv(df):
    direction = np.where(df['close_price'] > df['close_price'].shift(1), 1,
                         np.where(df['close_price'] < df['close_price'].shift(1), -1, 0))
    volume_change = df['volume'] * direction
    obv = volume_change.cumsum()
    obv[:1] = np.nan  # أول قيمة NaN
    return obv



def calculate_volume_roc(df, period=20):
    vol = df['volume']
    vol_roc = ((vol - vol.shift(period)) / vol.shift(period)) * 100
    vol_roc[:period] = np.nan
    return vol_roc


def calculate_cmf(df, period=20):
    mfv = ((df['close_price'] - df['low_price']) - (df['high_price'] - df['close_price'])) / (df['high_price'] - df['low_price'])
    mfv = mfv.fillna(0)  # لتجنب القسمة على صفر
    mfv *= df['volume']
    cmf = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    cmf[:period] = np.nan
    return cmf






def calculate_vwap(df):
    typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
    cum_vol_tp = (typical_price * df['volume']).cumsum()
    cum_vol = df['volume'].cumsum()
    vwap = cum_vol_tp / cum_vol
    return vwap





















    FUTURE_OFFSET = 2
    FRACTAL_OFFSET = 2


def add_raw_columns_and_masks(df, df_indicators):


    # الأعمدة الأساسية (الأسعار والحجم)
    df_indicators['open_time'] = df['open_time']
    df_indicators['open_price'] = df['open_price']
    df_indicators['high_price'] = df['high_price']
    df_indicators['low_price'] = df['low_price']
    df_indicators['close_price'] = df['close_price']
    df_indicators['volume'] = df['volume']

    # RSI
    df_indicators['RSI'] = calculate_rsi(df)
    df_indicators['RSI_mask'] = df_indicators['RSI'].isna().astype(int)

    # Stochastic
    stoch_k, stoch_d = calculate_stoch(df)
    df_indicators['Stoch_K'] = stoch_k
    df_indicators['Stoch_D'] = stoch_d
    df_indicators['Stoch_K_mask'] = stoch_k.isna().astype(int)
    df_indicators['Stoch_D_mask'] = stoch_d.isna().astype(int)

    # MACD
    macd_line, macd_signal, macd_hist = calculate_macd(df)
    df_indicators['MACD_Line'] = macd_line
    df_indicators['MACD_Signal'] = macd_signal
    df_indicators['MACD_Hist'] = macd_hist
    df_indicators['MACD_Line_mask'] = macd_line.isna().astype(int)
    df_indicators['MACD_Signal_mask'] = macd_signal.isna().astype(int)
    df_indicators['MACD_Hist_mask'] = macd_hist.isna().astype(int)

    # AO
    ao = calculate_ao(df)
    df_indicators['AO'] = ao
    df_indicators['AO_mask'] = ao.isna().astype(int)

    # CCI
    cci = calculate_cci(df)
    df_indicators['CCI'] = cci
    df_indicators['CCI_mask'] = cci.isna().astype(int)

    # MFI
    mfi = calculate_mfi(df)
    df_indicators['MFI'] = mfi
    df_indicators['MFI_mask'] = mfi.isna().astype(int)

    # SMA 20
    sma_20 = calculate_sma(df)
    df_indicators['SMA_20'] = sma_20
    df_indicators['SMA_20_mask'] = sma_20.isna().astype(int)

    # EMA 20
    ema_20 = calculate_ema(df)
    df_indicators['EMA_20'] = ema_20
    df_indicators['EMA_20_mask'] = ema_20.isna().astype(int)

    # ADX
    adx = calculate_adx(df)
    df_indicators['ADX'] = adx
    df_indicators['ADX_mask'] = adx.isna().astype(int)



    FUTURE_OFFSET_ICHIMOKU = 26

    tenkan, kijun, span_a, span_b, chikou = calculate_ichimoku(df)

    df_indicators['Ichimoku_Tenkan'] = tenkan
    df_indicators['Ichimoku_Kijun'] = kijun

    df_indicators['Ichimoku_SpanA'] = span_a.shift(FUTURE_OFFSET_ICHIMOKU)
    df_indicators['Ichimoku_SpanB'] = span_b.shift(FUTURE_OFFSET_ICHIMOKU)
    df_indicators['Ichimoku_Chikou'] = chikou.shift(-FUTURE_OFFSET_ICHIMOKU)  # chikou إزاحة للخلف (متأخر للأمام)

    df_indicators['Ichimoku_Tenkan_mask'] = tenkan.isna().astype(int)
    df_indicators['Ichimoku_Kijun_mask'] = kijun.isna().astype(int)
    df_indicators['Ichimoku_SpanA_mask'] = df_indicators['Ichimoku_SpanA'].isna().astype(int)
    df_indicators['Ichimoku_SpanB_mask'] = df_indicators['Ichimoku_SpanB'].isna().astype(int)
    df_indicators['Ichimoku_Chikou_mask'] = df_indicators['Ichimoku_Chikou'].isna().astype(int)




    # ATR و NATR
    atr = calculate_atr(df)
    natr = calculate_natr(df)
    df_indicators['ATR'] = atr
    df_indicators['NATR'] = natr
    df_indicators['ATR_mask'] = atr.isna().astype(int)
    df_indicators['NATR_mask'] = natr.isna().astype(int)

    # SAR
    sar = calculate_sar(df)
    df_indicators['SAR'] = sar
    df_indicators['SAR_mask'] = sar.isna().astype(int)

    # SuperTrend
    supertrend = calculate_supertrend(df)
    df_indicators['SuperTrend'] = supertrend
    df_indicators['SuperTrend_mask'] = supertrend.isna().astype(int)

    # Bollinger Bands
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(df)
    df_indicators['BB_Upper'] = bb_upper
    df_indicators['BB_Lower'] = bb_lower
    df_indicators['BB_Middle'] = bb_middle
    df_indicators['BB_Upper_mask'] = bb_upper.isna().astype(int)
    df_indicators['BB_Lower_mask'] = bb_lower.isna().astype(int)
    df_indicators['BB_Middle_mask'] = bb_middle.isna().astype(int)

    # Donchian Channels
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(df)
    df_indicators['Donchian_Upper'] = donchian_upper
    df_indicators['Donchian_Lower'] = donchian_lower
    df_indicators['Donchian_Middle'] = donchian_middle
    df_indicators['Donchian_Upper_mask'] = donchian_upper.isna().astype(int)
    df_indicators['Donchian_Lower_mask'] = donchian_lower.isna().astype(int)
    df_indicators['Donchian_Middle_mask'] = donchian_middle.isna().astype(int)

    # Keltner Channels
    keltner_upper, keltner_lower, keltner_middle = calculate_keltner(df)
    df_indicators['Keltner_Upper'] = keltner_upper
    df_indicators['Keltner_Lower'] = keltner_lower
    df_indicators['Keltner_Middle'] = keltner_middle
    df_indicators['Keltner_Upper_mask'] = keltner_upper.isna().astype(int)
    df_indicators['Keltner_Lower_mask'] = keltner_lower.isna().astype(int)
    df_indicators['Keltner_Middle_mask'] = keltner_middle.isna().astype(int)

    # STC
    stc = calculate_stc(df)
    df_indicators['STC'] = stc
    df_indicators['STC_mask'] = stc.isna().astype(int)

    # Pivot Points
    pivot, r1, r2, s1, s2 = calculate_pivot_points(df)
    df_indicators['Pivot'] = pivot
    df_indicators['R1'] = r1
    df_indicators['R2'] = r2
    df_indicators['S1'] = s1
    df_indicators['S2'] = s2
    df_indicators['Pivot_mask'] = pivot.isna().astype(int)
    df_indicators['R1_mask'] = r1.isna().astype(int)
    df_indicators['R2_mask'] = r2.isna().astype(int)
    df_indicators['S1_mask'] = s1.isna().astype(int)
    df_indicators['S2_mask'] = s2.isna().astype(int)

    # Fibonacci Levels
    fib_0_0, fib_0_236, fib_0_382, fib_0_5, fib_0_618 = calculate_fibonacci_levels(df)
    df_indicators['Fib_0_0'] = fib_0_0
    df_indicators['Fib_0_236'] = fib_0_236
    df_indicators['Fib_0_382'] = fib_0_382
    df_indicators['Fib_0_5'] = fib_0_5
    df_indicators['Fib_0_618'] = fib_0_618
    df_indicators['Fib_0_0_mask'] = fib_0_0.isna().astype(int)
    df_indicators['Fib_0_236_mask'] = fib_0_236.isna().astype(int)
    df_indicators['Fib_0_382_mask'] = fib_0_382.isna().astype(int)
    df_indicators['Fib_0_5_mask'] = fib_0_5.isna().astype(int)
    df_indicators['Fib_0_618_mask'] = fib_0_618.isna().astype(int)

    # Fib 1.0
    fib_1_0 = calculate_fib_1_0(df)
    df_indicators['Fib_1_0'] = fib_1_0
    df_indicators['Fib_1_0_mask'] = fib_1_0.isna().astype(int)








# STC
    stc = calculate_stc(df).shift(FUTURE_OFFSET)
    df_indicators['STC'] = stc
    df_indicators['STC_mask'] = stc.isna().astype(int)






    # Price Channel High (shifted)
    price_channel_high = calculate_price_channel_high(df).shift(FRACTAL_OFFSET)
    df_indicators['PriceChannel_High'] = price_channel_high
    df_indicators['PriceChannel_High_mask'] = price_channel_high.isna().astype(int)

    # Price Channel Low (shifted)
    price_channel_low = calculate_price_channel_low(df).shift(FRACTAL_OFFSET)
    df_indicators['PriceChannel_Low'] = price_channel_low
    df_indicators['PriceChannel_Low_mask'] = price_channel_low.isna().astype(int)

    # حجم التداول: التغير النسبي
    df_indicators['Volume_Change_Pct'] = df['volume'].pct_change() * 100

    # OBV
    obv = calculate_obv(df)
    df_indicators['OBV'] = obv
    df_indicators['OBV_mask'] = obv.isna().astype(int)

    # Volume ROC
    vol_roc = calculate_volume_roc(df)
    df_indicators['Volume_ROC'] = vol_roc
    df_indicators['Volume_ROC_mask'] = vol_roc.isna().astype(int)

    # CMF
    cmf = calculate_cmf(df)
    df_indicators['CMF'] = cmf
    df_indicators['CMF_mask'] = cmf.isna().astype(int)

    # VWAP
    vwap = calculate_vwap(df)
    df_indicators['VWAP'] = vwap
    df_indicators['VWAP_mask'] = vwap.isna().astype(int)

    return df_indicators



# ==========================
# المؤشرات الفنية
# ==========================
def calculate_all_indicators_with_fallback(df):
    df_out = pd.DataFrame(index=df.index)
    df_out = add_raw_columns_and_masks(df, df_out)
    return df_out









# ==========================
# Main loop
# ==========================
def main():
    print("🚀 بدء التحليل الزمني لفريم 5 دقائق")
    while True:
        try:
            symbols = get_symbols()
            for symbol in symbols:
                df = load_and_validate_data(symbol)
                if df is None:
                    continue

                df_ind = calculate_all_indicators_with_fallback(df)



    #            if len(df_ind) > 50:
    #              df_ind = df_ind.iloc[50:]
     #           else:
     #               continue

                save_to_db(symbol, df_ind)




            print(f"✅ الدورة اكتملت عند {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")

        except Exception as e:
            print(f"❌ خطأ أثناء المعالجة: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
