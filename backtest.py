import pandas as pd
import numpy as np


def run_backtest_engine(df, n_shares, tp, sl, comision=0.00125):
    """Motor de backtest con soporte Long/Short y comisiones."""
    cash = 1_000_000.0
    position = 0  # 1: Long, -1: Short
    entry_price = 0
    portfolio_values = []

    for i in range(len(df)):
        current_price = df.iloc[i]['Close']

        # Señales (Confirmación 2 de 3)
        rsi = df.iloc[i]['RSI']
        macd = df.iloc[i]['MACD']
        macd_s = df.iloc[i]['MACD_signal']
        bbl, bbu = df.iloc[i]['BBL'], df.iloc[i]['BBU']

        # Condiciones relajadas para dar margen a la optimización
        buy_signal = (int(rsi < 45) + int(macd > macd_s) + int(current_price < bbl * 1.01)) >= 2
        sell_signal = (int(rsi > 55) + int(macd < macd_s) + int(current_price > bbu * 0.99)) >= 2

        # 1. GESTIÓN DE SALIDAS
        if position != 0:
            if position == 1:
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            if pnl_pct >= tp or pnl_pct <= -sl:
                order_value = n_shares * current_price
                fee = order_value * comision
                if position == 1:
                    cash += order_value - fee
                else:
                    profit = (entry_price - current_price) * n_shares
                    cash += (n_shares * entry_price) + profit - fee
                position, entry_price = 0, 0

        # 2. GESTIÓN DE ENTRADAS
        if position == 0:
            if buy_signal:
                cost = n_shares * current_price
                fee = cost * comision
                if cash >= (cost + fee):
                    cash -= (cost + fee)
                    position, entry_price = 1, current_price
            elif sell_signal:
                cost = n_shares * current_price
                fee = cost * comision
                if cash >= (cost + fee):
                    cash -= (cost + fee)
                    position, entry_price = -1, current_price

        # 3. VALORIZACIÓN
        if position == 1:
            val = cash + (n_shares * current_price)
        elif position == -1:
            profit = (entry_price - current_price) * n_shares
            val = cash + (n_shares * entry_price) + profit
        else:
            val = cash
        portfolio_values.append(val)

    return portfolio_values


def calculate_metrics(portfolio_values):
    p_series = pd.Series(portfolio_values)
    returns = p_series.pct_change().dropna()
    total_return = (p_series.iloc[-1] - p_series.iloc[0]) / p_series.iloc[0]
    max_dd = abs(((p_series - p_series.cummax()) / p_series.cummax()).min())

    # Anualización para velas de 5 min
    ann_factor = np.sqrt(288 * 365)
    sharpe = (returns.mean() / returns.std() * ann_factor) if returns.std() != 0 else -1

    return {
        "final_value": p_series.iloc[-1],
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "win_rate": (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    }