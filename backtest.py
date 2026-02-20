import pandas as pd
import numpy as np
import ta


def run_backtest_engine(data, params):
    """
    Ejecuta la simulación de trading y retorna el historial del portafolio.
    """
    cash = 1_000_000
    initial_cash = cash
    COM = 0.125 / 100
    active_longs = []
    active_shorts = []
    portfolio_history = [cash]
    returns = []
    trades = []  # 1 para acierto, 0 para pérdida

    # Configuración de indicadores
    df = data.copy()
    df["rsi"] = ta.momentum.RSIIndicator(df.Close, window=params['rsi_window']).rsi()
    macd_ind = ta.trend.MACD(df.Close, window_slow=26, window_fast=12, window_sign=9)
    df["macd_diff"] = macd_ind.macd_diff()
    bb_ind = ta.volatility.BollingerBands(df.Close, window=params['bb_window'], window_dev=params['bb_dev'])
    df["bb_low"] = bb_ind.bollinger_lband()
    df["bb_high"] = bb_ind.bollinger_hband()
    df = df.dropna()

    for _, row in df.iterrows():
        price = row.Close

        # --- GESTIÓN DE POSICIONES ABIERTAS ---
        # Longs: Cierre por TP/SL
        for pos in active_longs.copy():
            current_val = price * pos['shares']
            if current_val <= pos['sl'] or current_val >= pos['tp']:
                cash += current_val * (1 - COM)
                trades.append(1 if current_val > pos['entry_val'] else 0)
                active_longs.remove(pos)

        # Shorts: Cierre por TP/SL (Sin apalancamiento)
        for pos in active_shorts.copy():
            current_liab = price * pos['shares']
            # En Short: SL si sube, TP si baja
            if current_liab >= pos['sl'] or current_liab <= pos['tp']:
                pnl = (pos['entry_price'] - price) * pos['shares']
                cash += pos['collateral'] + pnl - (current_liab * COM)
                trades.append(1 if pnl > 0 else 0)
                active_shorts.remove(pos)

        # --- GENERACIÓN DE SEÑALES (2 DE 3) ---
        # Alcista: RSI bajo, MACD subiendo, BB tocando fondo
        long_cond = (int(row.rsi < params['rsi_lower']) +
                     int(row.macd_diff > 0) +
                     int(price < row.bb_low)) >= 2

        # Bajista: RSI alto, MACD bajando, BB tocando techo
        short_cond = (int(row.rsi > params['rsi_upper']) +
                      int(row.macd_diff < 0) +
                      int(price > row.bb_high)) >= 2

        # --- EJECUCIÓN ---
        n_shares = params['n_shares']
        position_value = price * n_shares

        if long_cond and cash >= position_value * (1 + COM):
            cash -= position_value * (1 + COM)
            active_longs.append({
                'entry_price': price, 'entry_val': position_value, 'shares': n_shares,
                'sl': position_value * (1 - params['stop_loss']),
                'tp': position_value * (1 + params['take_profit'])
            })

        elif short_cond and cash >= position_value * (1 + COM):
            # Bloqueamos el colateral 1:1 (No leverage)
            cash -= position_value * (1 + COM)
            active_shorts.append({
                'entry_price': price, 'shares': n_shares, 'collateral': position_value,
                'sl': position_value * (1 + params['stop_loss']),
                'tp': position_value * (1 - params['take_profit'])
            })

        # --- VALORIZACIÓN DEL PORTAFOLIO ---
        val_long = sum([price * p['shares'] for p in active_longs])
        val_short = sum([p['collateral'] + (p['entry_price'] - price) * p['shares'] for p in active_shorts])
        current_total = cash + val_long + val_short

        returns.append((current_total / portfolio_history[-1]) - 1)
        portfolio_history.append(current_total)

    return pd.Series(portfolio_history), pd.Series(returns), trades


def calculate_metrics(history, returns, trades):
    """ Retorna el diccionario de métricas exigido por el reporte """
    if len(returns) < 5: return {"Calmar": 0, "Sharpe": 0, "Sortino": 0}

    total_ret = (history.iloc[-1] / history.iloc[0]) - 1
    # Anualización para 5 minutos
    ann_factor = 105120
    ann_ret = (1 + total_ret) ** (ann_factor / len(returns)) - 1

    # Drawdown
    cum_max = history.cummax()
    drawdown = (cum_max - history) / cum_max
    mdd = drawdown.max()

    # Ratios
    sharpe = (returns.mean() / returns.std()) * np.sqrt(ann_factor) if returns.std() != 0 else 0
    neg_ret = returns[returns < 0]
    sortino = (returns.mean() / neg_ret.std()) * np.sqrt(ann_factor) if len(neg_ret) > 0 else 0
    calmar = ann_ret / mdd if mdd > 0 else 0

    win_rate = (sum(trades) / len(trades)) * 100 if len(trades) > 0 else 0

    return {
        "Annualized Return": ann_ret,
        "Max Drawdown": mdd,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Calmar Ratio": calmar,
        "Win Rate (%)": win_rate,
        "Total Trades": len(trades)
    }


def objective(data, trial):
    """ Interfaz para Optuna """
    params = {
        'n_shares': trial.suggest_float("n_shares", 0.1, 5.0),
        'take_profit': trial.suggest_float("take_profit", 0.01, 0.15),
        'stop_loss': trial.suggest_float("stop_loss", 0.01, 0.15),
        'rsi_window': trial.suggest_int("rsi_window", 7, 30),
        'rsi_lower': trial.suggest_int("rsi_lower", 20, 40),
        'rsi_upper': trial.suggest_int("rsi_upper", 60, 80),
        'bb_window': trial.suggest_int("bb_window", 14, 50),
        'bb_dev': trial.suggest_float("bb_dev", 1.5, 2.5)
    }
    hist, ret, trd = run_backtest_engine(data, params)
    met = calculate_metrics(hist, ret, trd)
    return met["Calmar Ratio"]