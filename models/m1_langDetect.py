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
def extract_mfcc(audio_path, n_mfcc=13):
    waveform, sample_rate = torchaudio.load(audio_path)
    transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    )
    mfcc = transform(waveform)  # (channel, n_mfcc, time)
    mfcc = mfcc.mean(dim=0, keepdim=True)  # make it 1 channel
    return mfcc  # shape: [1, n_mfcc, time]

# ========== 3. Regularized CNN Model ==========
class LanguageDetectionCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(LanguageDetectionCNN, self).__init__()
        # Input shape: [batch_size, 1, 13, time]
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),  # Added BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),  # Add dropout to conv layers
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),  # Added BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),  # Add dropout to conv layers
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),  # Added BatchNorm
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 10))
        )
        
        # Fixed size after AdaptiveAvgPool2d: [batch_size, 64*2*10]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*2*10, 128),
            nn.BatchNorm1d(128),  # Added BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),  # Added an extra layer with smaller size
            nn.BatchNorm1d(64),   # Added BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ========== 4. Data Augmentation ==========
def augment_mfcc(mfcc, max_len=300):
    """Apply augmentations to MFCC features"""
    batch_size, channels, n_mfcc, time = mfcc.shape
    
    # List of possible augmentations
    augmentations = []
    
    # 1. Time masking - mask random time steps
    if random.random() > 0.5:
        mask_size = random.randint(5, min(30, time // 4))
        mask_start = random.randint(0, time - mask_size)
        time_mask = mfcc.clone()
        time_mask[:, :, :, mask_start:mask_start+mask_size] = 0
        augmentations.append(time_mask)
    
    # 2. Frequency masking - mask random frequency bands
    if random.random() > 0.5:
        mask_size = random.randint(1, min(3, n_mfcc // 2))
        mask_start = random.randint(0, n_mfcc - mask_size)
        freq_mask = mfcc.clone()
        freq_mask[:, :, mask_start:mask_start+mask_size, :] = 0
        augmentations.append(freq_mask)
    
    # 3. Noise addition - add small random noise
    if random.random() > 0.5:
        noise = torch.randn_like(mfcc) * 0.05  # small noise
        noise_added = mfcc + noise
        augmentations.append(noise_added)
    
    # If no augmentations were applied, return original
    if not augmentations:
        return mfcc
    
    # Randomly select one of the augmentations
    return random.choice(augmentations)

# ========== 5. Collate Function with Augmentation ==========
def collate_fn_with_augmentation(batch, max_len=300, augment=True):
    mfccs = []
    labels = []
    
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
    
    # Apply augmentation only during training
    if augment:
        mfccs = augment_mfcc(mfccs, max_len)
        
    labels = torch.stack(labels).unsqueeze(1)
    return mfccs, labels

# ========== 6. Label Noise Function ==========
def add_label_noise(labels, noise_probability=0.1):
    """Randomly flip labels with a certain probability"""
    # Create a mask of labels to flip
    mask = torch.rand_like(labels) < noise_probability
    # Flip the selected labels (0->1, 1->0)
    noisy_labels = labels.clone()
    noisy_labels[mask] = 1 - noisy_labels[mask]
    return noisy_labels

# ========== 7. Training Function with Label Noise ==========
def train_language_detection(model, data_loader, criterion, optimizer, device, epoch, label_noise_prob=0.05):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (mfccs, labels) in enumerate(data_loader):
        mfccs, labels = mfccs.to(device), labels.to(device)
        
        # Optionally add label noise
        if label_noise_prob > 0:
            training_labels = add_label_noise(labels, label_noise_prob)
        else:
            training_labels = labels

        # Forward pass
        outputs = model(mfccs)
        loss = criterion(outputs, training_labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Optional: Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics (using original labels for accuracy calculation)
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (i+1) % 5 == 0:  # Print every 5 batches
            print(f'Epoch [{epoch+1}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}')
    
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

# ========== 9. Prediction Function ==========
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

# ========== 10. Simplified Model for Lower Capacity ==========
class SimpleLanguageDetectionCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SimpleLanguageDetectionCNN, self).__init__()
        # Input shape: [batch_size, 1, n_mfcc, time]
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, stride=1, padding=1),  # Fewer filters
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1),  # Fewer filters
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d((2, 10))  # Simplified, removed one conv layer
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24*2*10, 64),  # Smaller hidden layer
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Add this function to your existing m1_langDetect.py file:

def detect_language(audio_path):
    """
    Detects whether the audio is in English or Thai language.
    
    Args:
        audio_path (str): Path to the audio file to analyze
        
    Returns:
        str: Detected language ("english" or "thai")
    """
    # Check if file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model_path = "final_model.pth"  # Try the final model first
    if not os.path.exists(model_path):
        model_path = "best_model.pth"  # Fall back to best model if final not found
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found. Please ensure either final_model.pth or best_model.pth exists.")
    
    # Initialize model
    model = LanguageDetectionCNN(dropout_rate=0.5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Use the existing predict_audio function
    language, _ = predict_audio(model, audio_path, device)
    
    # Return lowercase language name to match the expected format
    return language.lower()

# ========== 11. Main Training Script ==========
if __name__ == "__main__":
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    dataset_path = "dataset"  # Replace with your dataset path
    batch_size = 8  # Increased batch size
    num_epochs = 30  # Increased epochs since we'll use early stopping
    learning_rate = 0.0005  # Reduced learning rate
    weight_decay = 1e-3  # Increased weight decay for L2 regularization
    label_noise = 0.05  # 5% probability of flipping labels
    use_simple_model = False  # Set to True if you need even lower accuracy
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Set up dataset and data loaders
        dataset = LanguageDataset(dataset_path)
        
        # Perform train/validation/test split
        train_size = int(0.7 * len(dataset))  # Reduced training set size
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], 
            generator=torch.Generator().manual_seed(seed)  # Ensure reproducible splits
        )
        
        # Use augmentation only for training
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=lambda batch: collate_fn_with_augmentation(batch, augment=True), 
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=lambda batch: collate_fn_with_augmentation(batch, augment=False), 
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=lambda batch: collate_fn_with_augmentation(batch, augment=False), 
            num_workers=0
        )
        
        # Initialize model (choose between regular and simple model)
        if use_simple_model:
            model = SimpleLanguageDetectionCNN(dropout_rate=0.5).to(device)
        else:
            model = LanguageDetectionCNN(dropout_rate=0.5).to(device)
            
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(  # Changed to AdamW
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # More aggressive LR scheduling
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=3, factor=0.3
        )
        # We'll manually print when learning rate changes
        
        print(f"Model architecture:\n{model}")
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience = 7
        counter = 0
        best_model_path = "best_model.pth"
        
        # For tracking target accuracy models
        target_models = []
        
        # Training loop
        for epoch in range(num_epochs):
            # Train with augmented data and label noise
            train_loss, train_acc = train_language_detection(
                model, train_loader, criterion, optimizer, device, epoch, 
                label_noise_prob=label_noise
            )
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
            
            # Validate
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
            
            # Target accuracy range check (80-90%)
            if 80.0 <= val_acc <= 90.0:
                model_path = f"model_acc_{val_acc:.2f}_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), model_path)
                print(f"✅ Saved model in target accuracy range: {val_acc:.2f}% at epoch {epoch+1}")
                target_models.append((model_path, val_acc))
                
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # Save the model with the best validation loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved best model with validation loss: {val_loss:.4f}")
            else:
                counter += 1
            
            # Stop if no improvement for a certain number of epochs
            if counter >= patience:
                print(f"Early stopping triggered. Stopping at epoch {epoch+1}.")
                break
            
            # Learning rate scheduler step
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            # Print if learning rate changed
            if new_lr != old_lr:
                print(f"Learning rate reduced from {old_lr} to {new_lr}")
        
        print("Training complete.")
        
        # Select the best model in the target range if available
        if target_models:
            # Sort by accuracy (closer to 85% is better)
            target_models.sort(key=lambda x: abs(x[1] - 85.0))
            best_target_model, best_target_acc = target_models[0]
            print(f"Best model in target range: {best_target_model} with {best_target_acc:.2f}% accuracy")
            
            # Load this model for testing
            model.load_state_dict(torch.load(best_target_model))
        else:
            # If no model in the target range, use the one with best validation loss
            print("No model found in target accuracy range. Using model with best validation loss.")
            model.load_state_dict(torch.load(best_model_path))
        
        # Final evaluation on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        
        if 80.0 <= test_acc <= 90.0:
            print(f"✅ SUCCESS! Final test accuracy {test_acc:.2f}% is in the target range (80-90%).")
        elif test_acc > 90.0:
            print(f"⚠️ Test accuracy {test_acc:.2f}% is ABOVE the target range.")
            print("Suggestions to decrease accuracy:")
            print("- Set use_simple_model = True")
            print("- Increase label_noise to 0.1 or higher")
            print("- Reduce n_mfcc in extract_mfcc to 8 or lower")
        else:  # < 80%
            print(f"⚠️ Test accuracy {test_acc:.2f}% is BELOW the target range.")
            print("Suggestions to increase accuracy:")
            print("- Set use_simple_model = False")
            print("- Decrease label_noise to 0.02 or lower")
            print("- Increase n_mfcc in extract_mfcc back to 13")
        
        # Save final model
        torch.save(model.state_dict(), "final_model.pth")
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