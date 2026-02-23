import optuna
import pandas as pd
from backtest import run_backtest_engine, calculate_metrics


def objective(trial, df):
    """
    Función objetivo optimizada para aprender de los errores y
    maximizar el rendimiento neto.
    """
    params = {
        # n_shares: Subimos el rango para que el capital de 1M trabaje de verdad
        "n_shares": trial.suggest_float("n_shares", 10.0, 30.0),

        # tp: Take Profit flexible (1% a 10%)
        "tp": trial.suggest_float("tp", 0.01, 0.10),

        # sl: Stop Loss estratégico (1% a 4%)
        "sl": trial.suggest_float("sl", 0.01, 0.04)
    }

    portfolio_values = run_backtest_engine(df, **params)
    m = calculate_metrics(portfolio_values)

    # REGLA DE ORO: Si no hay trades, penalizamos para obligarlo a operar
    if m["total_return"] == 0:
        return -5.0

    # Maximizamos el Retorno Total sumado a una fracción del Sharpe
    # Esto ayuda a que Optuna priorice ganar dinero pero con estabilidad
    score = m["total_return"] + (m["sharpe"] * 0.01)

    return score


def run_walk_forward_analysis(df, n_windows=4):
    """Implementación de WFA requerida por la rúbrica."""
    results = []
    window_size = len(df) // (n_windows + 1)

    for i in range(n_windows):
        train_start = i * (window_size // 2)
        train_end = train_start + window_size
        test_end = train_end + (window_size // 2)

        train_chunk = df.iloc[train_start:train_end]
        test_chunk = df.iloc[train_end:test_end]

        if len(test_chunk) < 50: break

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, train_chunk), n_trials=20)

        m = calculate_metrics(run_backtest_engine(test_chunk, **study.best_params))
        results.append({"window": i, "return": m["total_return"]})

    return results


def optimize_final_params(df):
    """Optimización final de 100 trials."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Usamos TPESampler (predeterminado) que es excelente para estos rangos
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, df), n_trials=100)

    return study.best_params