import os
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ========== 1. Dataset Class ==========
class LanguageDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label, lang_dir in enumerate(['english', 'thai']):
            folder = os.path.join(root_dir, lang_dir)
            if not os.path.exists(folder):
                print(f"Warning: {folder} does not exist")
                continue
            for file in os.listdir(folder):
                if file.endswith('.wav'):
                    self.samples.append((os.path.join(folder, file), label))
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return path, torch.tensor(label, dtype=torch.float)

# ========== 2. MFCC Extraction ==========
def extract_mfcc(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    )
    mfcc = transform(waveform)  # (channel, n_mfcc, time)
    mfcc = mfcc.mean(dim=0, keepdim=True)  # make it 1 channel
    return mfcc  # shape: [1, n_mfcc, time]

# ========== 3. Simplified CNN Model with Fixed Architecture ==========
class LanguageDetectionCNN(nn.Module):
    def __init__(self):
        super(LanguageDetectionCNN, self).__init__()
        # Input shape: [batch_size, 1, 13, time]
        # Convolutional layers with fixed dimensions
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Halves both dimensions
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Halves both dimensions again
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 10))  # Force output to fixed size regardless of input
        )
        
        # Fixed size after AdaptiveAvgPool2d: [batch_size, 64, 2, 10]
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Size becomes [batch_size, 64*2*10]
            nn.Linear(64*2*10, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ========== 4. Collate Function with Fixed-Length Padding ==========
def collate_fn(batch):
    mfccs = []
    labels = []
    
    # We'll use a fixed maximum length for consistency
    max_len = 300  # You can adjust this based on your dataset
    
    for path, label in batch:
        # Extract MFCC
        mfcc = extract_mfcc(path)
        
        # Handle case where mfcc is longer than max_len
        if mfcc.shape[-1] > max_len:
            # Truncate
            mfcc = mfcc[..., :max_len]
        else:
            # Pad
            pad_amount = max_len - mfcc.shape[-1]
            mfcc = F.pad(mfcc, (0, pad_amount))  # Pad only the time dimension
            
        mfccs.append(mfcc)
        labels.append(label)
        
    mfccs = torch.stack(mfccs)  # (B, 1, n_mfcc, time)
    labels = torch.stack(labels).unsqueeze(1)
    return mfccs, labels

# ========== 5. Training Function ==========
def train_language_detection(model, data_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (mfccs, labels) in enumerate(data_loader):
        mfccs, labels = mfccs.to(device), labels.to(device)

        # Forward pass
        outputs = model(mfccs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (i+1) % 5 == 0:  # Print every 5 batches
            print(f'Epoch [{epoch+1}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

# ========== 6. Evaluation Function ==========
def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for mfccs, labels in data_loader:
            mfccs, labels = mfccs.to(device), labels.to(device)
            outputs = model(mfccs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    return test_loss, accuracy

# ========== 7. Predict Single Audio ==========
def predict_audio(model, audio_path, device):
    model.eval()
    with torch.no_grad():
        mfcc = extract_mfcc(audio_path)
        
        # Make sure it's the right length
        max_len = 300  # Same as in collate_fn
        if mfcc.shape[-1] > max_len:
            mfcc = mfcc[..., :max_len]
        else:
            pad_amount = max_len - mfcc.shape[-1]
            mfcc = F.pad(mfcc, (0, pad_amount))
            
        mfcc = mfcc.unsqueeze(0).to(device)  # Add batch dimension
        output = model(mfcc)
        
        prediction = 1 if output.item() > 0.5 else 0
        confidence = output.item() if prediction == 1 else 1 - output.item()
        language = 'Thai' if prediction == 1 else 'English'
        
        return language, confidence

# ========== 8. Main Training Script ==========
if __name__ == "__main__":
    dataset_path = "dataset"  # Replace with your dataset path
    batch_size = 4
    num_epochs = 10
    learning_rate = 0.001
    
    # Force specific seed for reproducibility
    torch.manual_seed(42)
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up dataset and data loaders
    try:
        dataset = LanguageDataset(dataset_path)
        
        # Perform train/test split
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 collate_fn=collate_fn, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                collate_fn=collate_fn, num_workers=0)
        
        # Initialize model, loss, and optimizer
        model = LanguageDetectionCNN().to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
        
        print(f"Model architecture:\n{model}")
        
        # Train model
        print("Starting training...")
        best_accuracy = 0.0
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = train_language_detection(model, train_loader, criterion, optimizer, device, epoch)
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
            
            # Evaluate
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            print(f"Epoch {epoch+1}/{num_epochs} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
            
            # Update learning rate
            scheduler.step(test_loss)
            
            # Save best model
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(model.state_dict(), "best_language_detection_model.pth")
                print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
        
        print(f"Training complete. Best accuracy: {best_accuracy:.2f}%")
        
        # Final model save
        torch.save(model.state_dict(), "final_language_detection_model.pth")
        print("Final model saved.")
        
        # Test on a single audio file
        test_path = "sample_test.wav"  # Replace with your test file
        if os.path.exists(test_path):
            language, confidence = predict_audio(model, test_path, device)
            print(f"Prediction for {test_path}: {language} (Confidence: {confidence:.4f})")
        else:
            print(f"Test file {test_path} not found.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()