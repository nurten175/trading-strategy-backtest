"""
Financial Analysis & Backtesting

This script performs financial analysis and backtests a technical trading strategy
on historical price data.

How to run:
    python backtest.py

Requirements:
    pandas, numpy, matplotlib

Notes:
- Uses publicly available data (not included in this repo).
- Update the data input path in the CONFIG section below.
"""

# =========================
# AAPL - Teknik Analiz (CSV)
# MA + BB + MACD + RSI + VWMA
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# -------------------------
# 0) DATA (CSV'den oku)
# -------------------------
from pathlib import Path

csv_path = input("Enter full path to CSV file: ").strip()
csv_path = Path(csv_path)

historical_df = pd.read_csv(csv_path, sep=";")
historical_df.columns = [c.strip() for c in historical_df.columns]

historical_df = pd.read_csv(path, sep=";")
historical_df.columns = [c.strip() for c in historical_df.columns]

historical_df["Date"] = pd.to_datetime(historical_df["Date"], errors="coerce", dayfirst=True)
historical_df["Price"] = pd.to_numeric(
    historical_df["Price"].astype(str).str.replace(",", "", regex=False),
    errors="coerce"
)

# Volume kolonu (Vol. vs)
vol_candidates = [c for c in historical_df.columns if "vol" in c.lower()]
vol_col = vol_candidates[0] if len(vol_candidates) > 0 else None

def parse_volume(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper().replace(",", "")
    if s in ["", "N/A", "-"]:
        return np.nan
    mult = 1
    if s.endswith("K"):
        mult = 1_000; s = s[:-1]
    elif s.endswith("M"):
        mult = 1_000_000; s = s[:-1]
    elif s.endswith("B"):
        mult = 1_000_000_000; s = s[:-1]
    try:
        return float(s) * mult
    except:
        return np.nan

if vol_col is not None:
    historical_df[vol_col] = historical_df[vol_col].apply(parse_volume)

historical_df = historical_df.dropna(subset=["Date", "Price"]).sort_values("Date").set_index("Date")

# (Senin mantığın) verinin yarısı
historical_df = historical_df.iloc[:len(historical_df)//2].copy()
close_historical = historical_df["Price"]


# -------------------------
# 1) MOVING AVERAGE (MA)
# -------------------------
rolling_app5  = close_historical.rolling(window=5).mean()
rolling_app14 = close_historical.rolling(window=14).mean()
rolling_app21 = close_historical.rolling(window=21).mean()

MAs = pd.concat([close_historical, rolling_app5, rolling_app14, rolling_app21], axis=1)
MAs.columns = ["close", "short", "mid", "long"]
MAs.dropna(inplace=True)

def buy_sell_MA(MAs, opt=50):
    buy_sell = []
    buy_signal = []
    sell_signal = []
    flag = 42

    up = MAs[(np.array(MAs["short"] - MAs["mid"]) > 0) & (np.array(MAs["short"] - MAs["long"]) > 0)]
    down = MAs[(np.array(MAs["short"] - MAs["mid"]) < 0) & (np.array(MAs["short"] - MAs["long"]) < 0)]

    ups = np.percentile(np.array(up["short"] - up["mid"]), opt) if len(up) > 0 else 0
    downs = np.percentile(np.array(down["short"] - down["long"]), opt) if len(down) > 0 else 0

    for i in range(len(MAs)):
        if (MAs["short"].iloc[i] > MAs["mid"].iloc[i] + ups) and (MAs["short"].iloc[i] > MAs["long"].iloc[i] + ups):
            buy_signal.append(np.nan)
            if flag != 1:
                sell_signal.append(MAs["close"].iloc[i])
                buy_sell.append(MAs["close"].iloc[i])
                flag = 1
            else:
                sell_signal.append(np.nan)

        elif (MAs["short"].iloc[i] < MAs["mid"].iloc[i] + downs) and (MAs["short"].iloc[i] < MAs["long"].iloc[i] + downs):
            sell_signal.append(np.nan)
            if flag != 0:
                buy_signal.append(MAs["close"].iloc[i])
                buy_sell.append(-MAs["close"].iloc[i])
                flag = 0
            else:
                buy_signal.append(np.nan)

        else:
            buy_sell.append(np.nan)
            sell_signal.append(np.nan)
            buy_signal.append(np.nan)

    operations = np.array(buy_sell, dtype=float)
    operations = operations[~np.isnan(operations)]

    # PL hesabı (senin mantık)
    if len(operations) == 0:
        PL = 0.0
    else:
        neg = 0
        pos = 0
        for i in range(len(operations)):
            if operations[i] < 0:
                neg = i
                break
        for i in range(1, len(operations)):
            if operations[-i] > 0:
                pos = i if i == 1 else i - 1
                break
        operations = operations[neg:] if pos == 1 else (operations[neg:-pos] if pos != 0 else operations[neg:])
        PL = float(np.sum(operations))

    return buy_signal, sell_signal, PL

MAs["BUY_MA"]  = pd.Series(buy_sell_MA(MAs)[0], index=MAs.index)
MAs["SELL_MA"] = pd.Series(buy_sell_MA(MAs)[1], index=MAs.index)
print("MA PL:", buy_sell_MA(MAs)[2])


# -------------------------
# 2) BOLLINGER BANDS (BB)
# -------------------------
BBs = pd.DataFrame(index=MAs.index)
BBs["Close"] = MAs["close"]
BBs["SMA"] = BBs["Close"].rolling(window=20).mean()
BBs["STD"] = BBs["Close"].rolling(window=20).std()
BBs["upper"] = BBs["SMA"] + 2 * BBs["STD"]
BBs["lower"] = BBs["SMA"] - 2 * BBs["STD"]
BBs.dropna(inplace=True)

def buy_sell_BB(data):
    buy_sell = []
    buy_signal = []
    sell_signal = []
    flag = 42

    for i in range(len(data)):
        if data["Close"].iloc[i] > data["upper"].iloc[i]:
            buy_signal.append(np.nan)
            if flag != 1:
                sell_signal.append(data["Close"].iloc[i])
                buy_sell.append(data["Close"].iloc[i])
                flag = 1
            else:
                sell_signal.append(np.nan)

        elif data["Close"].iloc[i] < data["lower"].iloc[i]:
            sell_signal.append(np.nan)
            if flag != 0:
                buy_signal.append(data["Close"].iloc[i])
                buy_sell.append(-data["Close"].iloc[i])
                flag = 0
            else:
                buy_signal.append(np.nan)

        else:
            buy_sell.append(np.nan)
            sell_signal.append(np.nan)
            buy_signal.append(np.nan)

    operations = np.array(buy_sell, dtype=float)
    operations = operations[~np.isnan(operations)]

    if len(operations) == 0:
        PL = 0.0
    else:
        neg = 0
        pos = 0
        for i in range(len(operations)):
            if operations[i] < 0:
                neg = i
                break
        for i in range(1, len(operations)):
            if operations[-i] > 0:
                pos = i if i == 1 else i - 1
                break
        operations = operations[neg:] if pos == 1 else (operations[neg:-pos] if pos != 0 else operations[neg:])
        PL = float(np.sum(operations))

    return buy_signal, sell_signal, PL

BBs["BUY_BB"]  = pd.Series(buy_sell_BB(BBs)[0], index=BBs.index)
BBs["SELL_BB"] = pd.Series(buy_sell_BB(BBs)[1], index=BBs.index)
print("BB PL:", buy_sell_BB(BBs)[2])


# -------------------------
# 3) MACD
# -------------------------
MDs = pd.DataFrame(index=MAs.index)
MDs["close"] = MAs["close"]
MDs["short"] = MDs["close"].ewm(span=12, adjust=False).mean()
MDs["long"]  = MDs["close"].ewm(span=26, adjust=False).mean()
MDs["MACD"]  = MDs["short"] - MDs["long"]
MDs["signal"] = MDs["MACD"].ewm(span=9, adjust=False).mean()

def buy_sell_MD(data, opt=50):
    buy_sell = []
    buy_signal = []
    sell_signal = []
    flag = 42

    up = data[(np.array(data["MACD"] - data["signal"]) > 0)]
    down = data[(np.array(data["MACD"] - data["signal"]) < 0)]
    ups = np.percentile(np.array(up["MACD"] - up["signal"]), opt) if len(up) > 0 else 0
    downs = np.percentile(np.array(down["MACD"] - down["signal"]), opt) if len(down) > 0 else 0

    for i in range(len(data)):
        if data["MACD"].iloc[i] > data["signal"].iloc[i] + ups:
            buy_signal.append(np.nan)
            if flag != 1:
                sell_signal.append(data["close"].iloc[i])
                buy_sell.append(data["close"].iloc[i])
                flag = 1
            else:
                sell_signal.append(np.nan)

        elif data["MACD"].iloc[i] < data["signal"].iloc[i] + downs:
            sell_signal.append(np.nan)
            if flag != 0:
                buy_signal.append(data["close"].iloc[i])
                buy_sell.append(-data["close"].iloc[i])
                flag = 0
            else:
                buy_signal.append(np.nan)

        else:
            buy_sell.append(np.nan)
            sell_signal.append(np.nan)
            buy_signal.append(np.nan)

    operations = np.array(buy_sell, dtype=float)
    operations = operations[~np.isnan(operations)]

    if len(operations) == 0:
        PL = 0.0
    else:
        neg = 0
        pos = 0
        for i in range(len(operations)):
            if operations[i] < 0:
                neg = i
                break
        for i in range(1, len(operations)):
            if operations[-i] > 0:
                pos = i if i == 1 else i - 1
                break
        operations = operations[neg:] if pos == 1 else (operations[neg:-pos] if pos != 0 else operations[neg:])
        PL = float(np.sum(operations))

    return buy_signal, sell_signal, PL

MDs["BUY_MACD"]  = pd.Series(buy_sell_MD(MDs)[0], index=MDs.index)
MDs["SELL_MACD"] = pd.Series(buy_sell_MD(MDs)[1], index=MDs.index)
print("MACD PL:", buy_sell_MD(MDs)[2])


# -------------------------
# 4) RSI (DÜZELTİLMİŞ)
# -------------------------
RSs = pd.DataFrame(index=MAs.index)
RSs["close"] = MAs["close"]

RSs["Diff"] = RSs["close"].diff(1)
RSs["Gain"] = RSs["Diff"].where(RSs["Diff"] > 0, 0.0)
RSs["Loss"] = (-RSs["Diff"]).where(RSs["Diff"] < 0, 0.0)

RSs["Avg_Gain"] = RSs["Gain"].rolling(window=14).mean()
RSs["Avg_Loss"] = RSs["Loss"].rolling(window=14).mean()
rs = RSs["Avg_Gain"] / RSs["Avg_Loss"]
RSs["rsi"] = 100 - (100 / (1 + rs))
RSs.dropna(inplace=True)

def buy_sell_RS(RSs, opt_low=30, opt_high=70):
    buy_sell = []
    buy_signal = []
    sell_signal = []
    flag = 42

    for i in range(len(RSs)):
        if RSs["rsi"].iloc[i] > opt_high:
            buy_signal.append(np.nan)
            if flag != 1:
                sell_signal.append(RSs["close"].iloc[i])
                buy_sell.append(RSs["close"].iloc[i])
                flag = 1
            else:
                sell_signal.append(np.nan)

        elif RSs["rsi"].iloc[i] < opt_low:
            sell_signal.append(np.nan)
            if flag != 0:
                buy_signal.append(RSs["close"].iloc[i])
                buy_sell.append(-RSs["close"].iloc[i])
                flag = 0
            else:
                buy_signal.append(np.nan)

        else:
            buy_sell.append(np.nan)
            sell_signal.append(np.nan)
            buy_signal.append(np.nan)

    operations = np.array(buy_sell, dtype=float)
    operations = operations[~np.isnan(operations)]

    if len(operations) == 0:
        PL = 0.0
    else:
        neg = 0
        pos = 0
        for i in range(len(operations)):
            if operations[i] < 0:
                neg = i
                break
        for i in range(1, len(operations)):
            if operations[-i] > 0:
                pos = i if i == 1 else i - 1
                break
        operations = operations[neg:] if pos == 1 else (operations[neg:-pos] if pos != 0 else operations[neg:])
        PL = float(np.sum(operations))

    return buy_signal, sell_signal, PL

RSs["BUY_RSI"]  = pd.Series(buy_sell_RS(RSs)[0], index=RSs.index)
RSs["SELL_RSI"] = pd.Series(buy_sell_RS(RSs)[1], index=RSs.index)
print("RSI PL:", buy_sell_RS(RSs)[2])


# -------------------------
# 5) VWMA (Volume varsa)
# -------------------------
VWs = None
if vol_col is not None and vol_col in historical_df.columns:
    VWs = pd.DataFrame(index=historical_df.index)
    VWs["Close"] = historical_df["Price"]
    VWs["Volume"] = historical_df[vol_col]
    VWs["CXV"] = VWs["Close"] * VWs["Volume"]

    VWs["VW14"] = VWs["CXV"].rolling(14).sum() / VWs["Volume"].rolling(14).sum()
    VWs["VW21"] = VWs["CXV"].rolling(21).sum() / VWs["Volume"].rolling(21).sum()
    VWs["VW50"] = VWs["CXV"].rolling(50).sum() / VWs["Volume"].rolling(50).sum()
    VWs.dropna(inplace=True)

    def buy_sell_VW(data, opt=50):
        buy_sell = []
        buy_signal = []
        sell_signal = []
        flag = 42

        up = data[(np.array(data["VW14"] - data["VW21"]) > 0) & (np.array(data["VW14"] - data["VW50"]) > 0)]
        down = data[(np.array(data["VW14"] - data["VW21"]) < 0) & (np.array(data["VW14"] - data["VW50"]) < 0)]

        ups = np.percentile(np.array(up["VW14"] - up["VW21"]), opt) if len(up) > 0 else 0
        downs = np.percentile(np.array(down["VW14"] - down["VW50"]), opt) if len(down) > 0 else 0

        for i in range(len(data)):
            if (data["VW14"].iloc[i] > data["VW21"].iloc[i] + ups) and (data["VW14"].iloc[i] > data["VW50"].iloc[i] + ups):
                buy_signal.append(np.nan)
                if flag != 1:
                    sell_signal.append(data["Close"].iloc[i])
                    buy_sell.append(data["Close"].iloc[i])
                    flag = 1
                else:
                    sell_signal.append(np.nan)

            elif (data["VW14"].iloc[i] < data["VW21"].iloc[i] + downs) and (data["VW14"].iloc[i] < data["VW50"].iloc[i] + downs):
                sell_signal.append(np.nan)
                if flag != 0:
                    buy_signal.append(data["Close"].iloc[i])
                    buy_sell.append(-data["Close"].iloc[i])
                    flag = 0
                else:
                    buy_signal.append(np.nan)

            else:
                buy_sell.append(np.nan)
                sell_signal.append(np.nan)
                buy_signal.append(np.nan)

        operations = np.array(buy_sell, dtype=float)
        operations = operations[~np.isnan(operations)]

        if len(operations) == 0:
            PL = 0.0
        else:
            neg = 0
            pos = 0
            for i in range(len(operations)):
                if operations[i] < 0:
                    neg = i
                    break
            for i in range(1, len(operations)):
                if operations[-i] > 0:
                    pos = i if i == 1 else i - 1
                    break
            operations = operations[neg:] if pos == 1 else (operations[neg:-pos] if pos != 0 else operations[neg:])
            PL = float(np.sum(operations))

        return buy_signal, sell_signal, PL

    VWs["BUY_VW"]  = pd.Series(buy_sell_VW(VWs)[0], index=VWs.index)
    VWs["SELL_VW"] = pd.Series(buy_sell_VW(VWs)[1], index=VWs.index)
    print("VWMA PL:", buy_sell_VW(VWs)[2])

# -------------------------
# 6) DECISIONS (senin merge tarzın)
# -------------------------
decisions = pd.merge(BBs[["BUY_BB", "SELL_BB"]], MAs[["BUY_MA", "SELL_MA"]],
                     left_index=True, right_index=True, how="inner")
decisions = pd.merge(decisions, MDs[["BUY_MACD", "SELL_MACD"]],
                     left_index=True, right_index=True, how="inner")
decisions = pd.merge(decisions, RSs[["BUY_RSI", "SELL_RSI"]],
                     left_index=True, right_index=True, how="inner")

if VWs is not None:
    decisions = pd.merge(decisions, VWs[["BUY_VW", "SELL_VW"]],
                         left_index=True, right_index=True, how="inner")

print(decisions.dropna(thresh=1).head(30))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# BACKTEST HELPERS
# =========================

def _to_bool_signal(series):
    """NaN -> False, sayı varsa True"""
    return series.notna()

def make_strategy_signals(decisions):
    """
    decisions: kolonlar
      BUY_MA, SELL_MA, BUY_BB, SELL_BB (en az bunlar)
    Çıktı: dict of (buy_bool_series, sell_bool_series)
    """
    buy_ma  = _to_bool_signal(decisions["BUY_MA"])
    sell_ma = _to_bool_signal(decisions["SELL_MA"])

    buy_bb  = _to_bool_signal(decisions["BUY_BB"])
    sell_bb = _to_bool_signal(decisions["SELL_BB"])

    # Strateji-1: sadece MA
    S1_buy = buy_ma
    S1_sell = sell_ma

    # Strateji-2: MA + BB birlikte (aynı gün ikisi de sinyal verirse)
    S2_buy = buy_ma & buy_bb
    S2_sell = sell_ma & sell_bb

    return {
        "MA_only": (S1_buy, S1_sell),
        "MA_and_BB": (S2_buy, S2_sell),
    }

def backtest_long_only(close, buy_sig, sell_sig, initial_cash=10000.0, fee_rate=0.0):
    """
    Basit long-only backtest:
      - BUY sinyali gelince (pozisyon yoksa) tüm nakitle AL
      - SELL sinyali gelince (pozisyon varsa) SAT ve nakde geç
    fee_rate: işlem başına oransal komisyon (0.001 = binde 1 gibi)
    """
    close = close.dropna().copy()
    buy_sig = buy_sig.reindex(close.index).fillna(False)
    sell_sig = sell_sig.reindex(close.index).fillna(False)

    cash = initial_cash
    shares = 0.0

    equity = []
    trades = []  # (date, type, price, shares, cash_after)

    entry_price = None
    for dt in close.index:
        price = float(close.loc[dt])

        # önce SELL kontrol (aynı gün BUY+SELL gelirse önce SELL yapmak daha güvenli)
        if sell_sig.loc[dt] and shares > 0:
            # sat
            gross = shares * price
            fee = gross * fee_rate
            cash = gross - fee
            trades.append((dt, "SELL", price, shares, cash))
            shares = 0.0
            entry_price = None

        # sonra BUY
        if buy_sig.loc[dt] and shares == 0:
            # al (tüm nakitle)
            fee = cash * fee_rate
            buy_cash = cash - fee
            shares = buy_cash / price
            cash = 0.0
            trades.append((dt, "BUY", price, shares, cash))
            entry_price = price

        equity.append(cash + shares * price)

    equity = pd.Series(equity, index=close.index, name="equity")

    # metrikler
    total_return = (equity.iloc[-1] / initial_cash) - 1.0

    # buy&hold
    bh_return = (close.iloc[-1] / close.iloc[0]) - 1.0

    # max drawdown
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_dd = float(drawdown.min())

    # trade stats: win rate (BUY->SELL çiftleri)
    trades_df = pd.DataFrame(trades, columns=["date", "type", "price", "shares", "cash_after"])
    wins = 0
    losses = 0
    pnl_list = []

    if not trades_df.empty:
        # BUY ve SELL'i sırayla eşleştir
        open_buy = None
        for _, row in trades_df.iterrows():
            if row["type"] == "BUY":
                open_buy = row
            elif row["type"] == "SELL" and open_buy is not None:
                buy_price = float(open_buy["price"])
                sell_price = float(row["price"])
                pnl = (sell_price - buy_price) / buy_price
                pnl_list.append(pnl)
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                open_buy = None

    n_trades = wins + losses
    win_rate = (wins / n_trades) if n_trades > 0 else np.nan
    avg_trade_return = float(np.mean(pnl_list)) if len(pnl_list) > 0 else np.nan

    summary = {
        "final_equity": float(equity.iloc[-1]),
        "total_return_pct": float(total_return * 100),
        "buy_hold_return_pct": float(bh_return * 100),
        "max_drawdown_pct": float(max_dd * 100),
        "num_closed_trades": int(n_trades),
        "win_rate_pct": float(win_rate * 100) if not np.isnan(win_rate) else np.nan,
        "avg_trade_return_pct": float(avg_trade_return * 100) if not np.isnan(avg_trade_return) else np.nan,
    }

    return equity, trades_df, summary


# =========================
# RUN (decisions + close gerekli)
# =========================

# close serisi: MA tablosundan al (senin kodda MAs['close'] var)
close = MAs["close"].copy()

# decisions: senin merge ettiğin tablo (BUY_MA/SELL_MA/BUY_BB/SELL_BB kolonları olmalı)
strategies = make_strategy_signals(decisions)

results = {}
for name, (b, s) in strategies.items():
    eq, tr, summ = backtest_long_only(close, b, s, initial_cash=10000.0, fee_rate=0.0)
    results[name] = (eq, tr, summ)

# özet tablo
summary_table = pd.DataFrame({k: v[2] for k, v in results.items()}).T
print(summary_table)

# equity curve grafiği
plt.figure(figsize=(12, 5))
for name, (eq, _, _) in results.items():
    plt.plot(eq.index, eq.values, label=name)
plt.title("Equity Curves (Backtest)")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.legend()
plt.show()

