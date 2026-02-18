from data import load_data, preprocess_data
from optimization import optimize_backtest


#Entrypoint
def main():
    print("Starting...")
    data_train, data_test = load_data()

    data_train = preprocess_data(data_train)
    data_test = preprocess_data(data_test)

    best_params = optimize_backtest(data_train)
    print(best_params)

if __name__ == "__main__":
    main()
