import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import urllib.parse
from datetime import datetime
from sqlalchemy import create_engine, text
from utils.config import DB_CONN_STR, intervals
import traceback

SEQ_LENGTH = 30
MAX_LOOKBACK = 200
NEEDED_CANDLES = SEQ_LENGTH + MAX_LOOKBACK

NULL_MASK_COLUMNS = [
    "Stoch_D", "AO", "CCI", "SMA_20", "ADX", "Ichimoku_Kijun",
    "Ichimoku_SpanA", "Ichimoku_SpanB", "Ichimoku_Chikou",
    "BB_Upper", "BB_Lower", "BB_Middle",
    "Donchian_Upper", "Donchian_Lower", "Donchian_Middle",
    "Keltner_Upper", "Keltner_Lower",
    "Fib_0_0", "Fib_0_236", "Fib_0_382", "Fib_0_5", "Fib_0_618", "Fib_1_0",
    "Fractal_Up", "Fractal_Down", "PriceChannel_High", "PriceChannel_Low",
    "CMF"
]

def get_engine():
    conn_str_encoded = urllib.parse.quote_plus(DB_CONN_STR)
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={conn_str_encoded}")
    return engine

engine = get_engine()

def get_candle_data(symbol, timeframe):
    query = """
        SELECT open_time, open_price, high_price, low_price, close_price, volume
        FROM CandleData
        WHERE symbol = :symbol AND interval = :timeframe
        ORDER BY open_time
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params={"symbol": symbol, "timeframe": timeframe})
    return df



def calculate_indicators(df):
    result = df.copy()



# حساب المؤشرات لكل الشموع
def calculate_indicators(df):
    result = df.copy()

    # RSI
    delta = result['close_price'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    result['RSI'] = 100 - (100 / (1 + rs))

    # Stochastic
    k_period = 14
    d_period = 3
    low_min = result['low_price'].rolling(window=k_period).min()
    high_max = result['high_price'].rolling(window=k_period).max()
    result['Stoch_K'] = 100 * (result['close_price'] - low_min) / (high_max - low_min).replace(0, np.nan)
    result['Stoch_D'] = result['Stoch_K'].rolling(window=d_period).mean()

    # MACD
    ema_fast = result['close_price'].ewm(span=12, adjust=False).mean()
    ema_slow = result['close_price'].ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    result['MACD_Line'] = macd_line
    result['MACD_Signal'] = signal_line
    result['MACD_Hist'] = macd_line - signal_line

    # AO
    median_price = (result['high_price'] + result['low_price']) / 2
    sma5 = median_price.rolling(window=5).mean()
    sma34 = median_price.rolling(window=34).mean()
    result['AO'] = sma5 - sma34

    # CCI
    tp = (result['high_price'] + result['low_price'] + result['close_price']) / 3
    sma = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    result['CCI'] = (tp - sma) / (0.015 * mad)

    # MFI
    typical_price = tp
    money_flow = typical_price * result['volume']
    positive_flow = [0]
    negative_flow = [0]
    for i in range(1, len(typical_price)):
        if typical_price.iloc[i] > typical_price.iloc[i - 1]:
            positive_flow.append(money_flow.iloc[i])
            negative_flow.append(0)
        else:
            positive_flow.append(0)
            negative_flow.append(money_flow.iloc[i])
    pos_mf = pd.Series(positive_flow).rolling(window=14).sum()
    neg_mf = pd.Series(negative_flow).rolling(window=14).sum()
    result['MFI'] = 100 * (pos_mf / (pos_mf + neg_mf).replace(0, np.nan))





    # SMA و EMA
    result['SMA_20'] = result['close_price'].rolling(window=20).mean()
    result['EMA_20'] = result['close_price'].ewm(span=20, adjust=False).mean()

    # ADX
    high = result['high_price']
    low = result['low_price']
    close = result['close_price']
    plus_dm = high.diff().where(high.diff() > low.diff(), 0)
    minus_dm = low.diff().where(low.diff() > high.diff(), 0).abs()
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    result['ADX'] = dx.rolling(window=14).mean()

    # Ichimoku
    nine_high = high.rolling(window=9).max()
    nine_low = low.rolling(window=9).min()
    result['Ichimoku_Tenkan'] = (nine_high + nine_low) / 2

    period26_high = high.rolling(window=26).max()
    period26_low = low.rolling(window=26).min()
    result['Ichimoku_Kijun'] = (period26_high + period26_low) / 2

    result['Ichimoku_SpanA'] = ((result['Ichimoku_Tenkan'] + result['Ichimoku_Kijun']) / 2).shift(26)
    period52_high = high.rolling(window=52).max()
    period52_low = low.rolling(window=52).min()
    result['Ichimoku_SpanB'] = ((period52_high + period52_low) / 2).shift(26)
    result['Ichimoku_Chikou'] = close.shift(-26)

    # Parabolic SAR
    result['SAR'] = np.nan
    af = 0.02
    max_af = 0.2
    ep = 0
    trend_up = True
    sar = low.iloc[0]
    for i in range(2, len(result)):
        prev_close = close.iloc[i - 1]
        prev_sar = sar
        if trend_up:
            sar = prev_sar + af * (ep - prev_sar)
            sar = min(sar, low.iloc[i - 1], low.iloc[i - 2])
            if close.iloc[i] < sar:
                trend_up = False
                sar = ep
                ep = high.iloc[i]
                af = 0.02
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + 0.02, max_af)
        else:
            sar = prev_sar + af * (ep - prev_sar)
            sar = max(sar, high.iloc[i - 1], high.iloc[i - 2])
            if close.iloc[i] > sar:
                trend_up = True
                sar = ep
                ep = low.iloc[i]
                af = 0.02
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + 0.02, max_af)
        result.at[result.index[i], 'SAR'] = sar

    # SuperTrend
    atr = tr.rolling(window=10).mean()
    factor = 3.0
    hl2 = (high + low) / 2
    upperband = hl2 + (factor * atr)
    lowerband = hl2 - (factor * atr)
    supertrend = [np.nan] * len(result)
    direction = True  # True=uptrend, False=downtrend
    for i in range(1, len(result)):
        if close[i] > upperband[i - 1]:
            direction = True
        elif close[i] < lowerband[i - 1]:
            direction = False
        if direction:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]
    result['SuperTrend'] = supertrend



    # ATR (Average True Range)
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    result['ATR'] = tr.rolling(window=14).mean()

    # NATR (Normalized ATR)
    result['NATR'] = (result['ATR'] / close) * 100

    # Bollinger Bands
    sma_bb = close.rolling(window=20).mean()
    std_bb = close.rolling(window=20).std()
    result['BB_Upper'] = sma_bb + 2 * std_bb
    result['BB_Lower'] = sma_bb - 2 * std_bb
    result['BB_Middle'] = sma_bb

    # Donchian Channels
    result['Donchian_Upper'] = high.rolling(window=20).max()
    result['Donchian_Lower'] = low.rolling(window=20).min()
    result['Donchian_Middle'] = (result['Donchian_Upper'] + result['Donchian_Lower']) / 2

    # Keltner Channels
    ema_keltner = close.ewm(span=20, adjust=False).mean()
    atr_keltner = tr.rolling(window=20).mean()
    result['Keltner_Upper'] = ema_keltner + (2 * atr_keltner)
    result['Keltner_Lower'] = ema_keltner - (2 * atr_keltner)
    result['Keltner_Middle'] = ema_keltner

    # STC (Schaff Trend Cycle)
    try:
        macd = result['MACD_Line']
        stc_k_period = 10
        stc_d_period = 3
        stc_smooth_k = macd.rolling(window=stc_k_period).apply(
            lambda x: 100 * (x[-1] - min(x)) / (max(x) - min(x)) if max(x) != min(x) else 0,
            raw=True
        )
        stc_d = stc_smooth_k.rolling(window=stc_d_period).mean()
        result['STC'] = stc_d
    except Exception as e:
        print("❌ خطأ في حساب STC:", e)
        result['STC'] = np.nan

    # Pivot Points (تستند على شمعة اليوم السابق أو الإطار السابق)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    pivot = (prev_high + prev_low + prev_close) / 3
    result['Pivot'] = pivot
    result['R1'] = 2 * pivot - prev_low
    result['S1'] = 2 * pivot - prev_high
    result['R2'] = pivot + (prev_high - prev_low)
    result['S2'] = pivot - (prev_high - prev_low)



    # Fibonacci Retracement (نستخدم أعلى وأدنى سعر)
    fib_high = high.rolling(window=20).max()
    fib_low = low.rolling(window=20).min()
    diff = fib_high - fib_low
    result['Fib_0_0'] = fib_high
    result['Fib_0_236'] = fib_high - 0.236 * diff
    result['Fib_0_382'] = fib_high - 0.382 * diff
    result['Fib_0_5'] = fib_high - 0.5 * diff
    result['Fib_0_618'] = fib_high - 0.618 * diff
    result['Fib_1_0'] = fib_low



    # Fractals (باستخدام نافذة 5 شموع)
    result['Fractal_Up'] = result['high_price'].shift(2).where(
        (result['high_price'].shift(2) > result['high_price'].shift(0)) &
        (result['high_price'].shift(2) > result['high_price'].shift(1)) &
        (result['high_price'].shift(2) > result['high_price'].shift(3)) &
        (result['high_price'].shift(2) > result['high_price'].shift(4))
    )

    result['Fractal_Down'] = result['low_price'].shift(2).where(
        (result['low_price'].shift(2) < result['low_price'].shift(0)) &
        (result['low_price'].shift(2) < result['low_price'].shift(1)) &
        (result['low_price'].shift(2) < result['low_price'].shift(3)) &
        (result['low_price'].shift(2) < result['low_price'].shift(4))
    )


    # Price Channel (أعلى وأدنى سعر خلال آخر 20 شمعة)
    result['PriceChannel_High'] = high.rolling(window=20).max()
    result['PriceChannel_Low'] = low.rolling(window=20).min()




    # OBV (On-Balance Volume)
    result['OBV'] = 0
    result['OBV'] = np.where(result['close_price'] > result['close_price'].shift(1),
                             result['volume'],
                             np.where(result['close_price'] < result['close_price'].shift(1),
                                      -result['volume'], 0)).cumsum()

    # Volume ROC (Rate of Change of Volume)
    result['Volume_ROC'] = result['volume'].pct_change(periods=14) * 100

    # CMF (Chaikin Money Flow)
    mf_multiplier = ((result['close_price'] - result['low_price']) - (result['high_price'] - result['close_price'])) / \
                    (result['high_price'] - result['low_price']).replace(0, np.nan)
    mf_volume = mf_multiplier * result['volume']
    result['CMF'] = mf_volume.rolling(window=20).sum() / result['volume'].rolling(window=20).sum()

    # VWAP (Volume Weighted Average Price)
    cumulative_vp = (result['close_price'] * result['volume']).cumsum()
    cumulative_vol = result['volume'].cumsum()
    result['VWAP'] = cumulative_vp / cumulative_vol

    for col in NULL_MASK_COLUMNS:
        if col in result.columns:
            result[f"{col}_mask"] = result[col].isna().astype(int)
            result[col] = result[col].fillna(999999)


    return result













import time
from datetime import datetime
import pandas as pd
from sqlalchemy import text

# يفترض أنك قمت بتعريف engine و get_candle_data و calculate_indicators و NEEDED_CANDLES و NULL_MASK_COLUMNS في مكان آخر

def save_to_db(symbol, timeframe, df):
    query_existing = """
        SELECT open_time 
        FROM OscillatorPerCandle_5m
        WHERE symbol = :symbol AND interval = :interval
    """
    with engine.connect() as conn:
        existing = pd.read_sql(text(query_existing), conn, params={"symbol": symbol, "interval": timeframe})
    existing_open_times = set(existing['open_time'])

    records = []
    for _, row in df.iterrows():
        if pd.isna(row.get('RSI')) or row['open_time'] in existing_open_times:
            continue

        record = {
            "symbol": symbol,
            "interval": timeframe,
            "open_time": row['open_time'],
            "RSI": row.get('RSI'),
            "Stoch_K": row.get('Stoch_K'),
            "Stoch_D": row.get('Stoch_D'),
            "MACD_Line": row.get('MACD_Line'),
            "MACD_Signal": row.get('MACD_Signal'),
            "MACD_Hist": row.get('MACD_Hist'),
            "AO": row.get('AO'),
            "CCI": row.get('CCI'),
            "MFI": row.get('MFI'),
            "SMA_20": row.get('SMA_20'),
            "EMA_20": row.get('EMA_20'),
            "ADX": row.get('ADX'),
            "Ichimoku_Tenkan": row.get('Ichimoku_Tenkan'),
            "Ichimoku_Kijun": row.get('Ichimoku_Kijun'),
            "Ichimoku_SpanA": row.get('Ichimoku_SpanA'),
            "Ichimoku_SpanB": row.get('Ichimoku_SpanB'),
            "Ichimoku_Chikou": row.get('Ichimoku_Chikou'),
            "SAR": row.get('SAR'),
            "SuperTrend": row.get('SuperTrend'),
            "ATR": row.get('ATR'),
            "NATR": row.get('NATR'),
            "BB_Upper": row.get('BB_Upper'),
            "BB_Lower": row.get('BB_Lower'),
            "BB_Middle": row.get('BB_Middle'),
            "Donchian_Upper": row.get('Donchian_Upper'),
            "Donchian_Lower": row.get('Donchian_Lower'),
            "Donchian_Middle": row.get('Donchian_Middle'),
            "Keltner_Upper": row.get('Keltner_Upper'),
            "Keltner_Lower": row.get('Keltner_Lower'),
            "Keltner_Middle": row.get('Keltner_Middle'),
            "STC": row.get('STC'),
            "Pivot": row.get('Pivot'),
            "R1": row.get('R1'),
            "R2": row.get('R2'),
            "S1": row.get('S1'),
            "S2": row.get('S2'),
            "Fib_0_0": row.get('Fib_0_0'),
            "Fib_0_236": row.get('Fib_0_236'),
            "Fib_0_382": row.get('Fib_0_382'),
            "Fib_0_5": row.get('Fib_0_5'),
            "Fib_0_618": row.get('Fib_0_618'),
            "Fib_1_0": row.get('Fib_1_0'),
            "Fractal_Up": row.get('Fractal_Up'),
            "Fractal_Down": row.get('Fractal_Down'),
            "PriceChannel_High": row.get('PriceChannel_High'),
            "PriceChannel_Low": row.get('PriceChannel_Low'),
            "OBV": row.get('OBV'),
            "Volume_ROC": row.get('Volume_ROC'),
            "CMF": row.get('CMF'),
            "VWAP": row.get('VWAP'),
            "open_price": row.get('open_price'),
            "high_price": row.get('high_price'),
            "low_price": row.get('low_price'),
            "close_price": row.get('close_price'),
            "volume": row.get('volume'),
        }

        for col in NULL_MASK_COLUMNS:
            value = row.get(col)
            record[f"{col}_mask"] = int(round(value, 6) == 999999.0 if pd.notna(value) else False)

        records.append(record)

    if records:
        df_out = pd.DataFrame(records)
        with engine.begin() as conn:
            df_out.to_sql("OscillatorPerCandle_5m", con=conn, if_exists="append", index=False)

def get_symbols():
    query = "SELECT DISTINCT symbol FROM CandleData"
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df['symbol'].dropna().unique().tolist()

def get_intervals_to_process():
    return ['5m']

def main():
    print("\U0001F4E2 بدأ تنفيذ الملف oscillators_2.py")

    while True:
        try:
            print(f"\n⏰ الوقت الحالي: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
            intervals_to_process = get_intervals_to_process()
            if not intervals_to_process:
                print("⌛ لا توجد فريمات لتحليلها الآن.")
            else:
                print(f"🚀 الفريمات المطلوب تحليلها الآن: {intervals_to_process}")
                symbols = get_symbols()
                for symbol in symbols:
                    for tf in intervals_to_process:
                        print(f"🔄 تحليل {symbol} - {tf}")
                        try:
                            df = get_candle_data(symbol, tf)
                            if df is not None and len(df) >= NEEDED_CANDLES:
                                df_with_indicators = calculate_indicators(df)
                                save_to_db(symbol, tf, df_with_indicators)
                            else:
                                print(f"⚠️ بيانات غير كافية لـ {symbol} - {tf}")
                        except Exception as e:
                            print(f"❌ خطأ أثناء تحليل {symbol} - {tf}: {e}")
                            import traceback
                            traceback.print_exc()

        except Exception as outer_e:
            print(f"❌ خطأ عام في الدورة: {outer_e}")
            import traceback
            traceback.print_exc()

        time.sleep(60)

if __name__ == "__main__":
    main()
