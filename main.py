import pandas as pd
import numpy as np
from src.models.svr import run_svr
from src.training.train_lstm import run_lstm

def rmse_score(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def run_pl_prediction():
    # Load data
    df = pd.read_csv("data/FCPR-D1.csv")
    
    # Check if required columns exist
    if "PL" not in df.columns or "X" not in df.columns:
        raise ValueError("Required columns 'PL' and 'X' not found in the dataset")
    
    # Remove any NaN values
    df = df.dropna(subset=["PL", "X"])
    
    # Select the columns we want to use for the prediction
    all_X = df[["PL"]]
    all_y = df["X"]
    
    # Convert to numpy arrays for LSTM
    X_array = all_X.values
    y_array = all_y.values
    
    print(f"Dataset shape: {X_array.shape}")
    print(f"Number of samples: {len(X_array)}")
    
    # SVR
    print("\n" + "="*50)
    print("Running SVR...")
    y_test_svr, y_pred_svr = run_svr(all_X, all_y)
    svr_rmse = rmse_score(y_test_svr, y_pred_svr)
    print(f"SVR RMSE: {svr_rmse:.4f}")
    
    # LSTM
    print("\n" + "="*50)
    print("Running LSTM...")
    y_test_lstm, y_pred_lstm = run_lstm(X_array, y_array)
    lstm_rmse = rmse_score(y_test_lstm, y_pred_lstm)
    print(f"LSTM RMSE: {lstm_rmse:.4f}")
    
    # Compare results
    print("\n" + "="*50)
    print("Model Comparison:")
    print(f"SVR RMSE:  {svr_rmse:.4f}")
    print(f"LSTM RMSE: {lstm_rmse:.4f}")
    
    if svr_rmse < lstm_rmse:
        print(f"\nSVR performs better by {lstm_rmse - svr_rmse:.4f} RMSE")
    else:
        print(f"\nLSTM performs better by {svr_rmse - lstm_rmse:.4f} RMSE")
    
    # Return results dictionary
    results = {
        "svr": {
            "rmse": svr_rmse,
            "y_test": y_test_svr,
            "y_pred": y_pred_svr
        },
        "lstm": {
            "rmse": lstm_rmse,
            "y_test": y_test_lstm,
            "y_pred": y_pred_lstm
        }
    }
    
    return results


if __name__ == "__main__":
    try:
        results = run_pl_prediction()
        print("\nPrediction completed successfully!")
    except FileNotFoundError:
        print("Error: Could not find 'data/FCPR-D1.csv'. Please ensure the data file exists.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")