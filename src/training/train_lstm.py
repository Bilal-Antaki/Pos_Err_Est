from src.models.lstm import LSTMModel
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split

def run_lstm(X, y):
    # Convert to numpy arrays if they're lists
    X = np.array(X)
    y = np.array(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Create and train model
    model = LSTMModel()
    trainer = L.Trainer(max_epochs=300, log_every_n_steps=2)
    trainer.fit(model, train_dataloaders=train_dataloader)
    
    # Make predictions on test set
    model.eval()
    with torch.no_grad():
        y_pred = []
        for x in X_test_tensor:
            pred = model(x.unsqueeze(0))  # Add batch dimension
            y_pred.append(pred.item())
    
    y_pred = np.array(y_pred)
    
    return y_test, y_pred