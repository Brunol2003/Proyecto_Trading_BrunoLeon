import pandas as pd
import time
from data import load_data, preprocess_data
from optimization import run_walk_forward_analysis, optimize_final_params
from backtest import run_backtest_engine, calculate_metrics


def main():
    start_time = time.time()
    print("=" * 60)
    print("INICIANDO OPTIMIZACIÓN DE ESTRATEGIA BTC - PASO FINAL")
    print("=" * 60)

    # 1. Carga y Preprocesamiento
    raw_train, raw_test = load_data()
    if raw_train is None: return

    train_df = preprocess_data(raw_train)
    test_df = preprocess_data(raw_test)

    # 2. Walk-Forward (Requisito de Rúbrica)
    print("\n[1/3] Ejecutando Walk-Forward Analysis...")
    run_walk_forward_analysis(train_df)

    # 3. Optimización Final
    print("[2/3] Buscando mejores parámetros con Optuna (100 trials)...")
    best_params = optimize_final_params(train_df)
    print(f"\n> CONFIGURACIÓN GANADORA ENCONTRADA: {best_params}")

    # 4. Evaluación Out-of-Sample (Test)
    print("\n[3/3] Aplicando parámetros al set de datos de TEST...")
    final_portfolio_values = run_backtest_engine(test_df, **best_params)
    m = calculate_metrics(final_portfolio_values)

    # REPORTE DE CONSOLA
    print("\n" + "╔" + "═" * 45 + "╗")
    print(f"║ {'RESULTADOS FINALES (TEST)':^43} ║")
    print("╠" + "═" * 45 + "╣")
    print(f"║ Retorno Total:         {m['total_return'] * 100:>18.2f}% ║")
    print(f"║ Sharpe Ratio:          {m['sharpe']:>18.4f} ║")
    print(f"║ Max Drawdown:          {m['max_drawdown'] * 100:>18.2f}% ║")
    print(f"║ Valor Final:           ${m['final_value']:>17,.2f} ║")
    print(f"║ Win Rate:              {m['win_rate'] * 100:>18.2f}% ║")
    print("╚" + "═" * 45 + "╝")

    # Guardar para el visualizador
    resultado_df = pd.DataFrame({
        "Datetime": test_df["Datetime"],
        "Portfolio_Value": final_portfolio_values
    })
    resultado_df.to_csv("resultado_portfolio_test.csv", index=False)

    exec_time = (time.time() - start_time) / 60
    print(f"\n✅ Proceso completado en {exec_time:.2f} minutos.")
    print("👉 Ahora puedes ejecutar visualizer.py para ver la gráfica.")


if __name__ == "__main__":
    main()