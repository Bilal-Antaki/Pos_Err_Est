from src.models.lstm import LSTMModel
import lightning as L
from torch.utils.data import DataLoader
import torch

def run_lstm(X, y):
    model = LSTMModel()
    trainer = L.Trainer(max_epochs=300, log_every_n_steps=2)
    trainer.fit(model, train_dataloader=DataLoader)
    
    # Make predictions
    y_pred = model(torch.tensor(X)).detach()
    
    return y.values, y_pred.numpy()
