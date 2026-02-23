import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates


def generate_visual_report():
    # 1. CARGA DE DATOS
    try:
        df = pd.read_csv("resultado_portfolio_test.csv")
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
    except FileNotFoundError:
        print("❌ Error: No se encontró 'resultado_portfolio_test.csv'. Ejecuta el main.py primero.")
        return

    # 2. CONFIGURACIÓN DE ESTILO
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))

    # --- GRÁFICA DE VALOR DEL PORTAFOLIO ---
    ax.plot(df.index, df['Portfolio_Value'], color='#27ae60', linewidth=2, label='Valor del Portafolio (USD)')
    ax.fill_between(df.index, df['Portfolio_Value'], df['Portfolio_Value'].min(), color='#2ecc71', alpha=0.15)

    # Formateo de fechas para que se vean bien
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.xticks(rotation=45)

    # Títulos y etiquetas (Requerimientos de Rúbrica)
    plt.title('Performance del Portafolio: Mayo - Junio 2024', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Valor Total (USD)', fontsize=12)
    plt.xlabel('Fecha de Operación', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('portfolio_value_time.png', dpi=300)
    print("✅ Gráfica 'portfolio_value_time.png' guardada con éxito.")

    # --- GENERACIÓN DE TABLAS DE RETORNOS (REQ. RÚBRICA) ---
    # Resampleo usando las nuevas reglas de Pandas (ME = Month End, QE = Quarter End)
    monthly_ret = df['Portfolio_Value'].resample('ME').last().pct_change().fillna(0) * 100
    quarterly_ret = df['Portfolio_Value'].resample('QE').last().pct_change().fillna(0) * 100
    annual_ret = df['Portfolio_Value'].resample('YE').last().pct_change().fillna(0) * 100

    print("\n" + "=" * 50)
    print("   TABLAS DE RETORNOS REQUERIDAS (MENSUAL/TRIM/ANUAL) ")
    print("=" * 50)

    print("\n[RETORNO MENSUAL]")
    for date, val in monthly_ret.items():
        print(f"{date.strftime('%B %Y')}: {val:+.4f}%")

    print("\n[RETORNO TRIMESTRAL]")
    for date, val in quarterly_ret.items():
        print(f"Q{date.quarter} {date.year}: {val:+.4f}%")

    print("\n[RETORNO ANUAL (PROYECTADO)]")
    for date, val in annual_ret.items():
        print(f"Año {date.year}: {val:+.4f}%")

    # Retorno total del periodo de prueba
    total_perf = ((df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0]) - 1) * 100
    print(f"\n[RETORNO TOTAL DEL TEST]: {total_perf:.4f}%")
    print("=" * 50)

    plt.show()


if __name__ == "__main__":
    generate_visual_report()