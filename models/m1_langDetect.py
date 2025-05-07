import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import os

class LanguageDetectionCNN(nn.Module):
    def __init__(self):
        super(LanguageDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 1)  # Output: 1 for binary classification

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def extract_mfcc(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    transform = T.MFCC(
        sample_rate=sample_rate,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
    )
    mfcc = transform(waveform)
    return mfcc.unsqueeze(0)  # Add batch dimension

def train_language_detection(model, data_loader, criterion, optimizer):
    model.train()
    for audio_path, label in data_loader:
        mfcc = extract_mfcc(audio_path)
        label = label.float().unsqueeze(1)
        
        # Forward pass
        output = model(mfcc)
        loss = criterion(output, label)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Example usage:
# model = LanguageDetectionCNN()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.BCELoss()
