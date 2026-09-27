import torch
import torch.nn as nn

class MusicLSTM(nn.Module):
    def __init__(self, n_vocab, hidden_dim=256, n_layers=2, dropout=0.3):
        super(MusicLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(1, hidden_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_vocab)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        out = self.fc(out)
        return out
