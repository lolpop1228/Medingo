import torch
import torch.nn as nn
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Model definition (CNN)
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 1)  # Output: binary classification

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification
        return x

# Function to extract MFCC features
def extract_mfcc(audio_path, sr=16000, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.expand_dims(mfcc, axis=0)  # Add channel dimension
    return torch.tensor(mfcc, dtype=torch.float32)

# Dataset class
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        mfcc = extract_mfcc(audio_path)
        return mfcc, label

# Example training loop (simplified)
def train(model, dataloader, epochs=10):
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

# Sample usage: Load data and train
# file_paths = [list of paths to .wav files]
# labels = [list of 0 or 1 indicating the language]
# dataset = AudioDataset(file_paths, labels)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# model = AudioCNN()
# train(model, dataloader)
