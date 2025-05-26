from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def rmse_score(y_true, y_pred):
    """Calculate Root Mean Square Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def run_pl_prediction():
    df = pd.read_csv("data/FCPR-D1.csv")
    
    # select the columns we want to use for the prediction
    all_X = df[["PL"]]
    all_y = df["X"]
    print(all_X.head())
    print("-"*80)
    print(all_y.head())
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.2, random_state=42)

    # Using SVR for regression
    regressor = svm.SVR()
    regressor.fit(X_train, y_train)

    # Calculate RMSE score
    y_pred = regressor.predict(X_test)
    score = rmse_score(y_test, y_pred)
    print(f"RMSE Score: {score:.4f}")
    return score


if __name__ == "__main__":
    results = run_pl_prediction()