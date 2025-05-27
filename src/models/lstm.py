import torch
import torch.nn as nn
from torch.optim import Adam
import lightning as L

class LSTMModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        # Simple neural network instead of LSTM for this single-feature regression
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        # Ensure input is properly shaped: (batch_size, n_features)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Make sure we have the right number of features
        if x.shape[-1] != 1:
            x = x.view(-1, 1)  # Reshape to (batch_size, 1)
            
        return self.network(x).squeeze(-1)  # Remove last dimension for scalar output
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)  # Lower learning rate
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        
        # Ensure shapes match
        if y_pred.shape != y.shape:
            y_pred = y_pred.view_as(y)
            
        loss = nn.MSELoss()(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
