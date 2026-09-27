import argparse
import sys
from src.train import train_model
from src.generate import generate_music
from src.preprocess import process_midi_dataset
import music21
import os

def prepare(args):
    print("Preparing dataset...")
    os.makedirs("data/midi", exist_ok=True)
    if args.demo:
        print("Downloading demo MIDI files from music21 corpus...")
        paths = music21.corpus.getComposer('bach')
        for i, p in enumerate(paths[:3]):
            try:
                s = music21.converter.parse(p)
                out_path = os.path.join("data", "midi", f"demo_bach_{i}.mid")
                s.write('midi', fp=out_path)
                print(f"Saved {out_path}")
            except Exception as e:
                print(f"Failed to save {p}: {e}")
                
    if args.train:
        train_model(epochs=args.epochs, batch_size=args.batch_size, sequence_length=args.seq_len)

def train(args):
    train_model(epochs=args.epochs, batch_size=args.batch_size, sequence_length=args.seq_len)

def generate(args):
    generate_music(length=args.length, temperature=args.temperature)

def main():
    parser = argparse.ArgumentParser(description="CodeAlpha Music Generation Project")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Prepare command
    parser_prep = subparsers.add_parser("prepare", help="Prepare MIDI dataset")
    parser_prep.add_argument("--demo", action="store_true", help="Download demo bach corpus to data/midi")
    parser_prep.add_argument("--train", action="store_true", help="Train after preparing")
    parser_prep.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser_prep.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser_prep.add_argument("--seq-len", type=int, default=100, help="Sequence length")
    
    # Train command
    parser_train = subparsers.add_parser("train", help="Train the LSTM model")
    parser_train.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser_train.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser_train.add_argument("--seq-len", type=int, default=100, help="Sequence length")
    
    # Generate command
    parser_gen = subparsers.add_parser("generate", help="Generate music using trained model")
    parser_gen.add_argument("--length", type=int, default=200, help="Number of notes to generate")
    parser_gen.add_argument("--temperature", type=float, default=0.8, help="Randomness in generation (higher = more random)")
    
    args = parser.parse_args()
    
    if args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        train(args)
    elif args.command == "generate":
        generate(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
