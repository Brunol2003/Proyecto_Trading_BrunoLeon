import pandas as pd
import pandas_ta as ta
import os


def load_data():
    """Carga archivos y limpia el gap de fechas del set de test."""
    train_path = os.path.join("data", "btc_project_train.csv")
    test_path = os.path.join("data", "btc_project_test.csv")

    try:
        if os.path.exists(train_path):
            data_train = pd.read_csv(train_path)
            data_test = pd.read_csv(test_path)
        else:
            data_train = pd.read_csv("btc_project_train.csv")
            data_test = pd.read_csv("btc_project_test.csv")

        data_train['Datetime'] = pd.to_datetime(data_train['Datetime'])
        data_test['Datetime'] = pd.to_datetime(data_test['Datetime'])

        # Filtro para evitar el hueco de enero a mayo en la gráfica final
        if data_test['Datetime'].min() < pd.Timestamp('2024-04-01'):
            data_test = data_test[data_test['Datetime'] >= '2024-05-01']

        return data_train.dropna(), data_test.dropna()
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return None, None


def preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Calcula indicadores técnicos estándar para BTC."""
    df = raw_data.copy().sort_values("Datetime")

    # Indicadores Clásicos
    df["RSI"] = ta.rsi(df["Close"], length=14)

    macd_df = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["MACD"] = macd_df.iloc[:, 0]
    df["MACD_signal"] = macd_df.iloc[:, 2]

    bbands = ta.bbands(df["Close"], length=20, std=2)
    df["BBL"] = bbands.iloc[:, 0]  # Banda Inferior
    df["BBU"] = bbands.iloc[:, 2]  # Banda Superior

    return df.dropna().reset_index(drop=True)