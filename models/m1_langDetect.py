import os
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
import numpy as np

# ========== 1. Dataset Class ==========
class LanguageDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label, lang_dir in enumerate(['english', 'thai']):
            folder = os.path.join(root_dir, lang_dir)
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if file.endswith('.wav'):
                        self.samples.append((os.path.join(folder, file), label))
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return path, torch.tensor(label, dtype=torch.float)

# ========== 2. Feature Extraction ==========
def extract_features(audio_path, n_mfcc=10):  # Reduced n_mfcc for less features
    waveform, sample_rate = torchaudio.load(audio_path)
    # Mix to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Extract MFCCs with simpler parameters
    transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 200, "n_mels": 16}  # Simplified mel parameters
    )
    mfcc = transform(waveform)  # (channel, n_mfcc, time)
    
    # Add basic normalization for better generalization
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
    
    return mfcc  # shape: [1, n_mfcc, time]

# ========== 3. Data Augmentation ==========
def augment_mfcc(mfcc, max_len=150):  # Reduced max length
    """Apply augmentations to MFCC features"""
    batch_size, channels, n_mfcc, time = mfcc.shape
    
    # List of possible augmentations
    augmentations = []
    
    # 1. Time masking - mask random time steps
    if random.random() > 0.5:
        mask_size = random.randint(5, min(20, time // 5))  # Smaller mask
        mask_start = random.randint(0, time - mask_size)
        time_mask = mfcc.clone()
        time_mask[:, :, :, mask_start:mask_start+mask_size] = 0
        augmentations.append(time_mask)
    
    # 2. Frequency masking - mask random frequency bands
    if random.random() > 0.5:
        mask_size = random.randint(1, min(2, n_mfcc // 3))  # Smaller mask
        mask_start = random.randint(0, n_mfcc - mask_size)
        freq_mask = mfcc.clone()
        freq_mask[:, :, mask_start:mask_start+mask_size, :] = 0
        augmentations.append(freq_mask)
    
    # 3. Noise addition - add small random noise
    if random.random() > 0.6:  # Reduced probability
        noise = torch.randn_like(mfcc) * 0.03  # Smaller noise
        noise_added = mfcc + noise
        augmentations.append(noise_added)
    
    # 4. Add time shifting
    if random.random() > 0.7:  # New augmentation
        shift = random.randint(-10, 10)
        shifted = torch.roll(mfcc, shifts=shift, dims=3)
        augmentations.append(shifted)
    
    # If no augmentations were applied, return original
    if not augmentations:
        return mfcc
    
    # Randomly select one of the augmentations
    return random.choice(augmentations)

# ========== 4. Balanced CNN Model ==========
class BalancedLanguageDetector(nn.Module):
    def __init__(self, dropout_rate=0.4):  # Increased dropout
        super(BalancedLanguageDetector, self).__init__()
        
        # Input shape: [batch_size, 1, n_mfcc, time]
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, stride=1, padding=1),  # Fewer filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate / 2),  # Dropout on feature maps
            
            nn.Conv2d(12, 20, kernel_size=3, stride=1, padding=1),  # Fewer filters
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate / 2)  # Dropout on feature maps
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 16))  # Fixed output size
        
        # Two fully connected layers with dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20 * 3 * 16, 48),  # Smaller hidden layer
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(48, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

# ========== 5. Collate Function with Augmentation ==========
def collate_fn(batch, max_len=150, augment=True):  # Reduced max length
    mfccs = []
    labels = []
    
    for path, label in batch:
        # Extract features
        try:
            mfcc = extract_features(path)
            
            # Handle variable length
            if mfcc.shape[-1] > max_len:
                # Random crop for training, center crop for testing
                if augment:
                    # Random crop
                    start = random.randint(0, mfcc.shape[-1] - max_len)
                    mfcc = mfcc[..., start:start+max_len]
                else:
                    # Center crop
                    start = (mfcc.shape[-1] - max_len) // 2
                    mfcc = mfcc[..., start:start+max_len]
            else:
                # Pad
                pad_amount = max_len - mfcc.shape[-1]
                mfcc = F.pad(mfcc, (0, pad_amount))
                
            mfccs.append(mfcc)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    if not mfccs:  # Handle empty batch
        return None, None
    
    mfccs = torch.stack(mfccs)
    
    # Apply augmentation only if specified
    if augment:
        mfccs = augment_mfcc(mfccs, max_len)
        
    labels = torch.stack(labels).unsqueeze(1)
    return mfccs, labels

# ========== 6. Add Label Noise Function ==========
def add_label_noise(labels, noise_probability=0.05):
    """Randomly flip labels with a certain probability to reduce overfitting"""
    mask = torch.rand_like(labels) < noise_probability
    noisy_labels = labels.clone()
    noisy_labels[mask] = 1 - noisy_labels[mask]
    return noisy_labels

# ========== 7. Training Function with Label Noise ==========
def train(model, data_loader, criterion, optimizer, device, label_noise_prob=0.05):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (mfccs, labels) in enumerate(data_loader):
        if mfccs is None:  # Skip empty batches
            continue
            
        mfccs, labels = mfccs.to(device), labels.to(device)
        
        # Add label noise to prevent overfitting
        if label_noise_prob > 0:
            training_labels = add_label_noise(labels, label_noise_prob)
        else:
            training_labels = labels
        
        # Forward pass
        outputs = model(mfccs)
        loss = criterion(outputs, training_labels)

        # L2 regularization via weight decay is handled by optimizer
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics (using original labels)
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    if total == 0:  # Handle empty data loader
        return 0, 0
        
    epoch_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

# ========== 8. Evaluation Function ==========
def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for mfccs, labels in data_loader:
            if mfccs is None:  # Skip empty batches
                continue
                
            mfccs, labels = mfccs.to(device), labels.to(device)
            outputs = model(mfccs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    if total == 0:  # Handle empty data loader
        return 0, 0
        
    test_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    return test_loss, accuracy

# ========== 9. Prediction Function ==========
def predict_audio(model, audio_path, device):
    model.eval()
    with torch.no_grad():
        try:
            mfcc = extract_features(audio_path)
            
            # Make sure it's the right length
            max_len = 150  # Same as in collate_fn
            if mfcc.shape[-1] > max_len:
                # Center crop for prediction
                start = (mfcc.shape[-1] - max_len) // 2
                mfcc = mfcc[..., start:start+max_len]
            else:
                # Pad
                pad_amount = max_len - mfcc.shape[-1]
                mfcc = F.pad(mfcc, (0, pad_amount))
                
            mfcc = mfcc.unsqueeze(0).to(device)  # Add batch dimension
            output = model(mfcc)
            
            prediction = 1 if output.item() > 0.5 else 0
            confidence = output.item() if prediction == 1 else 1 - output.item()
            language = 'Thai' if prediction == 1 else 'English'
            
            return language, confidence
        except Exception as e:
            print(f"Error predicting audio {audio_path}: {e}")
            return "Error", 0.0

# ========== 10. Main Script ==========
if __name__ == "__main__":
    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Configuration
    dataset_path = "dataset"  # Replace with your dataset path
    batch_size = 16
    num_epochs = 20
    learning_rate = 0.0005  # Lower learning rate
    weight_decay = 0.001  # L2 regularization
    label_noise = 0.05  # 5% probability of flipping labels
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Set up dataset and data loaders
        dataset = LanguageDataset(dataset_path)
        
        # Ensure dataset is not empty
        if len(dataset) == 0:
            raise ValueError("Dataset is empty. Please check your dataset path.")
        
        # Perform train/validation/test split
        train_size = int(0.6 * len(dataset))  # Reduced training set size
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], 
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Use augmentation only for training
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=lambda batch: collate_fn(batch, augment=True), 
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=lambda batch: collate_fn(batch, augment=False), 
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=lambda batch: collate_fn(batch, augment=False), 
            num_workers=0
        )
        
        # Initialize model
        model = BalancedLanguageDetector(dropout_rate=0.4).to(device)
        print(f"Model architecture:\n{model}")
        
        # Loss and optimizer with weight decay
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay  # L2 regularization
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=3, factor=0.5
        )
        
        # Best model tracking
        best_val_loss = float('inf')
        best_model_path = "best_balanced_model.pth"
        
        # Early stopping
        patience = 5
        counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            # Train with augmented data and label noise
            train_loss, train_acc = train(
                model, train_loader, criterion, optimizer, device, label_noise_prob=label_noise
            )
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
            
            # Validate
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
            
            # Check if we're in target accuracy range on validation set
            if 80.0 <= val_acc <= 90.0:
                print(f"✅ Validation accuracy {val_acc:.2f}% is in target range (80-90%)")
                
            # Learning rate scheduler
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model with validation loss: {val_loss:.4f}")
                counter = 0  # Reset early stopping counter
            else:
                counter += 1
                
            # Early stopping
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

            scheduler.step(val_loss)

            for param_group in optimizer.param_groups:
                print(f"Current learning rate: {param_group['lr']}")
        
        print("Training complete.")
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(best_model_path))
        
        # Evaluate on validation set again
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Best Model - Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        
        # Final evaluation on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        
        # Check if we achieved target accuracy on test set
        if 80.0 <= test_acc <= 90.0:
            print(f"✅ SUCCESS! Final test accuracy {test_acc:.2f}% is in the target range (80-90%).")
        elif test_acc > 90.0:
            print(f"⚠️ Test accuracy {test_acc:.2f}% is ABOVE the target range.")
        else:  # < 80%
            print(f"⚠️ Test accuracy {test_acc:.2f}% is BELOW the target range.")
        
        # Save final model
        torch.save(model.state_dict(), "final_balanced_model.pth")
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