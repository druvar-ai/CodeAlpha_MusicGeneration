import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from .midi_utils import get_notes_from_midi
import music21

class MIDIDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.long)

def prepare_sequences(notes, n_vocab, sequence_length=100):
    """
    Prepare the sequences used by the Neural Network.
    """
    # get all pitch names
    pitches = sorted(set(item for item in notes))
    
    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitches))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input_np = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # normalize input
    network_input_np = network_input_np / float(n_vocab)

    network_output_np = np.array(network_output)

    return network_input_np, network_output_np, note_to_int

def process_midi_dataset(data_dir, sequence_length=100):
    notes = []
    midi_files = glob.glob(os.path.join(data_dir, "*.mid")) + glob.glob(os.path.join(data_dir, "*.midi"))
    
    if not midi_files:
        print(f"No MIDI files found in {data_dir}. Generating some basic corpus pieces from music21...")
        paths = music21.corpus.getComposer('bach')
        midi_files = paths[:5]
        print(f"Using {len(midi_files)} Bach pieces from music21 corpus.")

    successful = 0
    skipped = 0
    
    for file in midi_files:
        file_notes = get_notes_from_midi(file)
        if file_notes is not None and len(file_notes) > 0:
            notes.extend(file_notes)
            successful += 1
        else:
            skipped += 1
            
    print(f"Processed files: {successful + skipped}")
    print(f"Successfully parsed: {successful}")
    print(f"Skipped/Failed: {skipped}")
    
    if len(notes) == 0:
        raise ValueError("No notes extracted. Please provide valid MIDI files.")
        
    n_vocab = len(set(notes))
    print(f"Total notes extracted: {len(notes)}")
    print(f"Vocabulary size: {n_vocab}")
    
    network_input, network_output, note_to_int = prepare_sequences(notes, n_vocab, sequence_length)
    print(f"Number of training sequences: {len(network_input)}")
    
    return network_input, network_output, note_to_int, n_vocab, notes
