#!/usr/bin/env python3
# nn_text_compressor.py
# A simple neural network-based text compressor using PyTorch
# Author: Jacek Witkowski ( jacek.witkowski1978@gmail .com )
# Date: 2025-12-17

# This programm is just the idea testing software.



import argparse
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm   


#################################################################
# Configuration 
NUMBER_OF_EPOCHS = 100 




################################################################
######
################################################################
class ByteTextDataset(Dataset):
    # wszystkie dane traningowe do wczytania jako dane do nauki
    # maybe a Oxford dictionary as an input?
    def __init__(self, files, seq_len=1024):

        self.data = bytearray()
        self.seq_len = seq_len
        for file in files:
            with open(file, 'rb') as f:
                self.data.extend(f.read())

        self.n = len(self.data)


    def __len__(self):
        return max(1, self.n // self.seq_len )
        #raise NotImplementedError

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = min(start + self.seq_len, self.n)
        part = self.data[start:end]
        x = torch.tensor(list(part[:-1]), dtype=torch.long) if len(part) > 1 else torch.tensor([0], dtype=torch.long)
        y = torch.tensor(list(part[1:]), dtype=torch.long) if len(part) > 1 else torch.tensor([0], dtype=torch.long)
        return x, y
    #raise NotImplementedError
    

class LSTMModel(nn.Module):
    def __init__(self, n_tokens=256, emb=128, hid=256, nlayers=2):
        super().__init__()
        self.emb = nn.Embedding(n_tokens, emb)
        self.lstm = nn.LSTM(emb, hid, nlayers, batch_first=True)
        self.fc = nn.Linear(hid, n_tokens)


    def forward(self, x, hidden=None):
        # x: (B, T)
        e = self.emb(x)
        out, hidden = self.lstm(e, hidden)
        logits = self.fc(out)
        return logits, hidden
    

def train_model(train_files, trained_model_path, liczbaEpok=NUMBER_OF_EPOCHS, batch_size=8, seq_len=1024, lr=1e-3):
    print(f"Training model with files: {train_files}")
    # Placeholder for training logic
    # Here you would implement the neural network training using PyTorch
    # and save the trained model to 'trained_model_path' if provided.
    if trained_model_path:
        print(f"Trained model will be saved to: {trained_model_path}")

        ds = ByteTextDataset(train_files, seq_len=seq_len)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        # Now we need a model
        model = LSTMModel().to('cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(liczbaEpok):
            model.train()
            total_loss = 0.0
            n_tokens = 0
            for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{liczbaEpok}"):
                x = x.to('cpu')
                y = y.to('cpu')
                logits, _ = model(x)
                B, T, C = logits.shape
                loss = criterion(logits.view(B*T, C), y.view(B*T))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * (B * T)
                n_tokens += B * T
            print(f"Epoch {epoch+1} Loss: {total_loss / n_tokens:.4f}")
            torch.save(model.state_dict(), trained_model_path)
        
        print("Training completed. Model was saved to ", trained_model_path)

    else:
        print("No path provided for saving the trained model.")


def main():
    arguments = argparse.ArgumentParser(description="Neural Network Text Compressor")
    arguments.add_argument("--mode", choices=["train", "compress", "decompress"], required=True, help="Mode of operation")
    arguments.add_argument("--train_files", nargs="*")
    arguments.add_argument("--infile")
    arguments.add_argument("--outfile")
    arguments.add_argument("--trained_model")

    args = arguments.parse_args()

    if args.mode == "train":
        if not args.train_files:
            print("Training files are required for training mode.\n")
            print("Usage: python3 nn_text_compressor.py --mode train --train_files file1.txt file2.txt\n")
            return
        train_model(args.train_files, args.trained_model)
    elif args.mode == "compress":
        if not args.infile or not args.outfile:
            print("Input and output files are required for compression mode.\n")
            print("Usage: python3 nn_text_compressor.py --mode compress --infile input.txt --outfile output.bin\n")
            return
        # Placeholder for compression logic
        print(f"Compressing {args.infile} to {args.outfile} (not implemented yet).")
    elif args.mode == "decompress":
        if not args.infile or not args.outfile:
            print("Input and output files are required for decompression mode.\n")
            print("Usage: python3 nn_text_compressor.py --mode decompress --infile input.bin --outfile output.txt\n")
            return
        # Placeholder for decompression logic
        print(f"Decompressing {args.infile} to {args.outfile} (not implemented yet).")
    else:
        print("Invalid mode. Valid modes are: train, compress, decompress.\n")
        print("Usage: python3 nn_text_compressor.py --mode [train|compress|decompress]\n")
        return

if __name__ == "__main__":
    main()