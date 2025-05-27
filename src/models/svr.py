from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np



def run_svr(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Using SVR for regression
    regressor = svm.SVR()
    regressor.fit(X_train, y_train)

    # Calculate RMSE score
    y_pred = regressor.predict(X_test)
    

    return y_test, y_pred