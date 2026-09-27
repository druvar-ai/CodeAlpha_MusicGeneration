import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .preprocess import process_midi_dataset, MIDIDataset
from .model import MusicLSTM

def train_model(data_dir="data/midi", model_dir="models", epochs=10, batch_size=64, sequence_length=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    os.makedirs(model_dir, exist_ok=True)
    
    print("Preparing dataset...")
    network_input, network_output, note_to_int, n_vocab, _ = process_midi_dataset(data_dir, sequence_length)
    
    dataset = MIDIDataset(network_input, network_output)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = MusicLSTM(n_vocab=n_vocab, hidden_dim=256, n_layers=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
    # Save model and metadata
    torch.save(model.state_dict(), os.path.join(model_dir, "music_lstm.pth"))
    metadata = {
        "n_vocab": n_vocab,
        "note_to_int": note_to_int,
        "sequence_length": sequence_length
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
        
    print("Training complete. Model and metadata saved.")
