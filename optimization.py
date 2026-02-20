import optuna
from backtest import objective, run_backtest_engine, calculate_metrics


def run_walk_forward_analysis(data):
    """
    Implementa el requisito:
    Train: 1 mes | Test: 1 semana | Step: 1 semana
    """
    # Aproximación de velas de 5 min:
    # 12 velas/hora * 24 horas * 30 días = 8640 velas (1 mes)
    # 12 velas/hour * 24 horas * 7 días = 2016 velas (1 semana)

    WINDOW_TRAIN = 8640
    WINDOW_TEST = 2016
    STEP = 2016  # Step forward semanal

    results = []
    start_idx = 0

    print(f"Iniciando Walk-Forward Analysis (Total filas: {len(data)})")

    # Mientras haya suficiente data para train + test
    while start_idx + WINDOW_TRAIN + WINDOW_TEST <= len(data):
        train_slice = data.iloc[start_idx: start_idx + WINDOW_TRAIN]
        test_slice = data.iloc[start_idx + WINDOW_TRAIN: start_idx + WINDOW_TRAIN + WINDOW_TEST]

        print(f"\n--- Optimizando ventana: {start_idx} a {start_idx + WINDOW_TRAIN} ---")

        study = optuna.create_study(direction='maximize')
        # El requerimiento pide entre 100-200 trials por ventana
        study.optimize(lambda trial: objective(train_slice, trial), n_trials=100)

        # Evaluar los mejores parámetros de esa ventana en su correspondiente semana de TEST
        best_params = study.best_params
        history, returns, trades = run_backtest_engine(test_slice, best_params)
        test_metrics = calculate_metrics(history, returns, trades)

        results.append({
            "window_start": start_idx,
            "best_params": best_params,
            "test_calmar": test_metrics["Calmar Ratio"]
        })

        start_idx += STEP

    return results


def optimize_final_params(train_data):
    """
    Optimización global sobre el archivo train completo para el reporte final.
    """
    print("\n[INFO] Ejecutando optimización final sobre btc_project_train.csv...")

    # Silenciamos los logs de Optuna para que la consola esté limpia
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(train_data, trial), n_trials=150)

    return study.best_params, study.best_value
