import pandas as pd
import numpy as np
from src.models.svr import run_svr
from src.training.train_lstm import run_lstm

def rmse_score(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def run_pl_prediction():
    df = pd.read_csv("data/FCPR-D1.csv")
    
    # select the columns we want to use for the prediction
    all_X = df[["PL"]]
    all_y = df["X"]
    X_list = all_X.values.tolist()
    y_list = all_y.values.tolist()

    # SVR
    y_test_svr, y_pred_svr = run_svr(all_X, all_y)
    svr_rmse = rmse_score(y_test_svr, y_pred_svr)
    print(f"SVR RMSE: {svr_rmse:.4f}")

    # LSTM
    y_test_lstm, y_pred_lstm = run_lstm(X_list, y_list)
    lstm_rmse = rmse_score(y_test_lstm, y_pred_lstm)
    print(f"LSTM RMSE: {lstm_rmse:.4f}")

if __name__ == "__main__":
    results = run_pl_prediction()