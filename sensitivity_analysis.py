import pandas as pd
from data import load_data, preprocess_data
from backtest import run_backtest_engine, calculate_metrics


def run_sensitivity_test():
    # 1. CARGA DE DATOS
    _, raw_test = load_data()
    df_test = preprocess_data(raw_test)

    # 2. DEFINE TUS MEJORES PARÁMETROS (Cópialos de lo que te dio el main.py)
    # Ejemplo: Si Optuna te dio tp=0.05, sl=0.02, n_shares=10.0
    optimal_params = {
        "n_shares": 10.0,  # <-- AJUSTA SEGÚN TU RESULTADO DEL MAIN
        "tp": 0.05,  # <-- AJUSTA SEGÚN TU RESULTADO DEL MAIN
        "sl": 0.02  # <-- AJUSTA SEGÚN TU RESULTADO DEL MAIN
    }

    results = []

    print("\n" + "=" * 60)
    print("   ANÁLISIS DE SENSIBILIDAD (PRUEBA DE ROBUSTEZ +/- 20%)")
    print("=" * 60)

    # Probamos: Original, -20% y +20% para cada parámetro clave
    for param_name in ["tp", "sl"]:
        for factor in [0.8, 1.0, 1.2]:  # -20%, Original, +20%
            test_params = optimal_params.copy()
            test_params[param_name] *= factor

            p_values = run_backtest_engine(df_test, **test_params)
            m = calculate_metrics(p_values)

            label = "ORIGINAL" if factor == 1.0 else f"{factor:+.1%}"
            results.append({
                "Parámetro": param_name.upper(),
                "Variación": label,
                "Valor": f"{test_params[param_name]:.4f}",
                "Retorno Total": f"{m['total_return'] * 100:.2f}%",
                "Max Drawdown": f"{m['max_drawdown'] * 100:.2f}%",
                "Sharpe": f"{m['sharpe']:.4f}"
            })

    # 3. MOSTRAR TABLA DE RESULTADOS
    sensitivity_df = pd.DataFrame(results)
    # Eliminar duplicados del valor 'ORIGINAL' para que la tabla sea limpia
    sensitivity_df = sensitivity_df.drop_duplicates(subset=['Parámetro', 'Variación'])

    print(sensitivity_df.to_string(index=False))
    print("\n" + "=" * 60)

    # 4. GUARDAR PARA EL REPORTE
    sensitivity_df.to_csv("sensitivity_results.csv", index=False)
    print("✅ Resultados guardados en 'sensitivity_results.csv'")


if __name__ == "__main__":
    run_sensitivity_test()