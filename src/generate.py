import os
import json
import numpy as np
import torch
import random
from .model import MusicLSTM
from .midi_utils import create_midi_from_notes
from .preprocess import process_midi_dataset

def generate_music(model_dir="models", output_dir="output", length=100, temperature=1.0, data_dir="data/midi"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating on device: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    metadata_path = os.path.join(model_dir, "metadata.json")
    model_path = os.path.join(model_dir, "music_lstm.pth")
    
    if not os.path.exists(metadata_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Model or metadata not found. Please train the model first.")
        
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    n_vocab = metadata["n_vocab"]
    note_to_int = metadata["note_to_int"]
    int_to_note = {v: k for k, v in note_to_int.items()}
    sequence_length = metadata["sequence_length"]
    
    model = MusicLSTM(n_vocab=n_vocab, hidden_dim=256, n_layers=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("Loading data for seed sequence...")
    # Get a random sequence from data for seed
    network_input, _, _, _, notes = process_midi_dataset(data_dir, sequence_length)
    
    if len(network_input) > 1:
        start = np.random.randint(0, len(network_input) - 1)
    else:
        start = 0
    pattern = network_input[start]
    
    prediction_output = []
    
    print(f"Generating {length} notes with temperature {temperature}...")
    
    with torch.no_grad():
        for note_index in range(length):
            prediction_input = torch.tensor(pattern, dtype=torch.float32).unsqueeze(0).to(device) # (1, seq_len, 1)
            
            output = model(prediction_input) # (1, n_vocab)
            output = output.squeeze(0) # (n_vocab)
            
            # Apply temperature
            output = output / temperature
            probs = torch.softmax(output, dim=0).cpu().numpy()
            
            # Sample from probabilities
            index = np.random.choice(len(probs), p=probs)
            
            result = int_to_note[index]
            prediction_output.append(result)
            
            # Update pattern
            new_input = np.array([[[index / float(n_vocab)]]])
            pattern = np.vstack((pattern[1:], new_input[0]))
            
    output_file = os.path.join(output_dir, "generated_music.mid")
    create_midi_from_notes(prediction_output, output_file)
    print(f"Generation complete. Saved to {output_file}")
