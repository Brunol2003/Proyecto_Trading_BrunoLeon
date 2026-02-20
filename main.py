import pandas as pd
import time
from data import load_data, preprocess_data
from optimization import run_walk_forward_analysis, optimize_final_params
from backtest import run_backtest_engine, calculate_metrics


def main():
    start_time = time.time()
    print("=" * 50)
    print("SISTEMA DE TRADING BTC/USDT - ESTRATEGIA MULTI-INDICADOR")
    print("=" * 50)

    # 1. CARGA Y PREPROCESAMIENTO
    print("\n[1/4] Cargando archivos CSV...")
    raw_train, raw_test = load_data()
    train_df = preprocess_data(raw_train)
    test_df = preprocess_data(raw_test)
    print(f"Líneas cargadas - Train: {len(train_df)} | Test: {len(test_df)}")

    # 2. WALK-FORWARD ANALYSIS (WFA)
    # Este paso es crítico para el reporte (demuestra robustez)
    print("\n[2/4] Iniciando Walk-Forward Analysis (Ventanas móviles)...")
    wf_results = run_walk_forward_analysis(train_df)

    # Mostrar resumen de WFA
    wf_calmars = [res['test_calmar'] for res in wf_results]
    avg_wf_calmar = sum(wf_calmars) / len(wf_calmars) if wf_calmars else 0
    print(f"\nResumen WFA: Calmar Promedio en Test = {avg_wf_calmar:.4f}")

    # 3. OPTIMIZACIÓN FINAL
    # Buscamos los parámetros definitivos usando todo el set de entrenamiento
    print("\n[3/4] Buscando mejores parámetros finales (150 trials)...")
    best_params, best_val = optimize_final_params(train_df)

    print("\n" + "-" * 30)
    print("MEJORES PARÁMETROS ENCONTRADOS:")
    for k, v in best_params.items():
        print(f" > {k}: {v}")
    print("-" * 30)

    # 4. EVALUACIÓN FINAL EN TEST (EL RESULTADO REAL)
    print("\n[4/4] Evaluando estrategia en 'btc_project_test.csv'...")
    history, returns, trades = run_backtest_engine(test_df, best_params)
    final_metrics = calculate_metrics(history, returns, trades)

    print("\n" + "!" * 30)
    print("RESULTADOS FINALES (DATA DE TEST):")
    for metric, value in final_metrics.items():
        if "Ratio" in metric or "Sharpe" in metric or "Sortino" in metric:
            print(f" - {metric}: {value:.4f}")
        else:
            print(f" - {metric}: {value:.2f}")
    print("!" * 30)

    # 5. GUARDAR RESULTADOS PARA EL REPORTE
    # Esto genera el CSV con el que harás la gráfica de "Portfolio Value"
    history.to_csv("resultado_portfolio_test.csv", index=False)

    end_time = time.time()
    total_min = (end_time - start_time) / 60
    print(f"\n[INFO] Tiempo total de ejecución: {total_min:.2f} minutos.")
    print("[INFO] Archivo 'resultado_portfolio_test.csv' generado para el reporte.")


if __name__ == "__main__":
    main()
