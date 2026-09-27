# CodeAlpha AI Internship — Task 3: Music Generation with AI

## 1. Project Overview
This project implements an AI-based music generation system that creates original music sequences using a Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN). 

## 2. CodeAlpha Requirement
- Collect and preprocess MIDI music data into note sequences.
- Build a deep learning model using an LSTM architecture.
- Train the model on the music dataset.
- Generate new sequences from the trained model and output them as a MIDI file.

## 3. Features
- **Data Preprocessing Pipeline**: Parses MIDI files extracting notes and chords, preparing sequences suitable for LSTM training.
- **LSTM Deep Learning Model**: Custom PyTorch LSTM that handles sequence prediction.
- **Configurable Training**: Simple CLI to configure epochs, sequence length, and batch sizes.
- **Temperature Sampling**: Allows introducing randomness during sequence generation to explore creative variations.
- **Automated Tests**: Unit tests verify the end-to-end functionality of data pipelines, model building, and music generation.

## 4. How the System Works
The AI learns temporal dependencies in musical sequences. It processes notes and chords by their pitch representations, learns the likelihood of the next note given the previous sequence, and iteratively generates notes from a random seed to form a new composition.

## 5. AI/ML Approach
- Deep Learning Framework: **PyTorch**
- Network: **LSTM (Long Short-Term Memory)**
- Optimization: **Adam Optimizer**
- Loss Function: **CrossEntropyLoss**

## 6. Dataset Preparation
You can provide your own MIDI files in the `data/midi/` directory or run the built-in preparation script which fetches some classical demo files (e.g., Bach) using `music21`.

## 7. Preprocessing
MIDI files are loaded using `music21`. Polyphonic elements (chords) are flattened or tokenized into numerical pitch combinations. The data is grouped into sequences of length `N` (e.g., 100), predicting the `N+1` note.

## 8. LSTM Architecture
The model features an input dimension of 1 (normalized token ID), passing through `N` stacked LSTM layers, a dropout layer to prevent overfitting, and a fully connected (Dense) layer to output a probability distribution across the entire musical vocabulary.

## 9. Training
To train the model on CPU or GPU automatically:
```bash
python main.py train --epochs 50 --batch-size 64
```
A smaller dataset or fewer epochs can be used for demonstration purposes.

## 10. Music Generation
Run the following to generate an original sequence based on learned patterns:
```bash
python main.py generate --length 200 --temperature 0.8
```

## 11. MIDI Output
The generated music is saved locally as a standard MIDI file located at: `output/generated_music.mid`

## 12. Optional Audio Playback/Conversion
The system outputs `.mid` files. These can be opened natively in most operating systems. If you prefer audio conversion (`.wav`), you can use tools like FluidSynth. However, keeping it lightweight, the core system does not strictly mandate audio packages.

## 13. Installation
Ensure you have Python 3.8+ installed.
```bash
pip install -r requirements.txt
```

## 14. Usage Commands
- **Prepare demo dataset**: `python main.py prepare --demo`
- **Train the model**: `python main.py train --epochs 10`
- **Generate music**: `python main.py generate --length 100`

## 15. Project Structure
```text
CodeAlpha_MusicGeneration/
├── data/midi/           # Place training .mid files here
├── models/              # Trained weights and vocabulary
├── output/              # Generated .mid outputs
├── src/                 # Source code (preprocessing, model, training, generation)
├── main.py              # CLI entrypoint
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── test_app.py          # Unit test suite
```

## 16. Testing
To run the automated tests verifying all functionalities:
```bash
python -m unittest test_app.py
```

## 17. Limitations
- Single Instrument: The basic tokenization primarily handles a single melodic track (Piano) efficiently.
- Musical Rhythm: Rhythm and velocity are simplified in this demonstration structure for the sake of the sequence prediction algorithm.

## 18. Future Improvements
- Multi-track support.
- Rhythm/duration modeling alongside pitch.
- Advanced architectures like Transformers.
