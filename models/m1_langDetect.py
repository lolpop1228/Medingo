import os
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn.metrics import classification_report, confusion_matrix
import librosa

# ========== 1. Enhanced Dataset Class with Balance Check ==========
class LanguageDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        language_counts = {'english': 0, 'thai': 0}
        
        for label, lang_dir in enumerate(['english', 'thai']):
            folder = os.path.join(root_dir, lang_dir)
            if not os.path.exists(folder):
                print(f"Warning: {folder} does not exist")
                continue
            for file in os.listdir(folder):
                if file.endswith('.wav'):
                    self.samples.append((os.path.join(folder, file), label))
                    language_counts[lang_dir] += 1
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"English samples: {language_counts['english']}")
        print(f"Thai samples: {language_counts['thai']}")
        
        # Check for severe imbalance
        total = sum(language_counts.values())
        if total > 0:
            eng_ratio = language_counts['english'] / total
            thai_ratio = language_counts['thai'] / total
            print(f"Dataset balance - English: {eng_ratio:.2%}, Thai: {thai_ratio:.2%}")
            
            if abs(eng_ratio - thai_ratio) > 0.3:  # More than 30% difference
                print("⚠️ WARNING: Severe class imbalance detected!")
                print("Consider balancing your dataset or using class weights.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return path, torch.tensor(label, dtype=torch.float)

# ========== 2. Robust MFCC Extraction ==========
def extract_mfcc_robust(audio_path, n_mfcc=13, target_sr=16000):
    """Enhanced MFCC extraction with better preprocessing"""
    try:
        # Load audio with librosa for better handling
        waveform, original_sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # Normalize audio to prevent clipping
        if np.max(np.abs(waveform)) > 0:
            waveform = waveform / np.max(np.abs(waveform)) * 0.9
        
        # Convert to torch tensor
        waveform = torch.from_numpy(waveform).unsqueeze(0)  # Add channel dimension
        
        # Apply pre-emphasis filter to balance frequency spectrum
        pre_emphasis = 0.97
        waveform = torch.cat([waveform[:, :1], waveform[:, 1:] - pre_emphasis * waveform[:, :-1]], dim=1)
        
        # Extract MFCC features
        transform = T.MFCC(
            sample_rate=target_sr,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": 512,  # Increased for better frequency resolution
                "hop_length": 256,  # Adjusted hop length
                "n_mels": 40,  # More mel filters
                "center": False,
                "f_min": 80,  # Set minimum frequency to avoid very low frequencies
                "f_max": target_sr // 2  # Nyquist frequency
            }
        )
        
        mfcc = transform(waveform)  # (channel, n_mfcc, time)
        
        # Apply delta and delta-delta features for more robust representation
        delta = torchaudio.transforms.ComputeDeltas()(mfcc)
        delta2 = torchaudio.transforms.ComputeDeltas()(delta)
        
        # Concatenate MFCC with deltas
        enhanced_mfcc = torch.cat([mfcc, delta, delta2], dim=1)  # Shape: [1, n_mfcc*3, time]
        
        # Normalize MFCC coefficients
        enhanced_mfcc = (enhanced_mfcc - enhanced_mfcc.mean(dim=-1, keepdim=True)) / (enhanced_mfcc.std(dim=-1, keepdim=True) + 1e-8)
        
        return enhanced_mfcc
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        # Return zero tensor as fallback
        return torch.zeros(1, n_mfcc * 3, 100)

# ========== 3. Improved CNN Model with Attention ==========
class ImprovedLanguageDetectionCNN(nn.Module):
    def __init__(self, n_mfcc=13, dropout_rate=0.3):
        super(ImprovedLanguageDetectionCNN, self).__init__()
        input_channels = n_mfcc * 3  # MFCC + delta + delta-delta
        
        # Feature extraction with residual connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Apply attention
        attention_weights = self.attention(x).unsqueeze(-1).unsqueeze(-1)
        x = x * attention_weights
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x

# ========== 4. Enhanced Data Augmentation ==========
def enhanced_augment_mfcc(mfcc, max_len=400):
    """More conservative augmentation to prevent overfitting"""
    batch_size, channels, n_mfcc, time = mfcc.shape
    
    # Apply augmentation with lower probability
    if random.random() > 0.7:  # Only 30% chance of augmentation
        augmentation_type = random.choice(['time_mask', 'freq_mask', 'noise'])
        
        if augmentation_type == 'time_mask':
            # Smaller time masks
            mask_size = random.randint(3, min(15, time // 8))
            mask_start = random.randint(0, time - mask_size)
            mfcc_aug = mfcc.clone()
            mfcc_aug[:, :, :, mask_start:mask_start+mask_size] *= 0.1  # Don't zero completely
            return mfcc_aug
            
        elif augmentation_type == 'freq_mask':
            # Smaller frequency masks
            mask_size = random.randint(1, min(2, n_mfcc // 4))
            mask_start = random.randint(0, n_mfcc - mask_size)
            mfcc_aug = mfcc.clone()
            mfcc_aug[:, :, mask_start:mask_start+mask_size, :] *= 0.1
            return mfcc_aug
            
        elif augmentation_type == 'noise':
            # Much smaller noise
            noise = torch.randn_like(mfcc) * 0.01
            return mfcc + noise
    
    return mfcc

# ========== 5. Enhanced Collate Function ==========
def enhanced_collate_fn(batch, max_len=400, augment=True):
    mfccs = []
    labels = []
    
    for path, label in batch:
        try:
            # Extract enhanced MFCC
            mfcc = extract_mfcc_robust(path)
            
            # Handle variable length sequences
            if mfcc.shape[-1] > max_len:
                # Random crop instead of just truncating from the beginning
                start_idx = random.randint(0, mfcc.shape[-1] - max_len) if augment else 0
                mfcc = mfcc[..., start_idx:start_idx + max_len]
            else:
                # Pad with zeros
                pad_amount = max_len - mfcc.shape[-1]
                mfcc = F.pad(mfcc, (0, pad_amount))
            
            mfccs.append(mfcc)
            labels.append(label)
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            # Skip this sample
            continue
    
    if not mfccs:  # If all samples failed
        return None, None
        
    mfccs = torch.stack(mfccs)
    
    # Apply conservative augmentation
    if augment:
        mfccs = enhanced_augment_mfcc(mfccs, max_len)
        
    labels = torch.stack(labels).unsqueeze(1)
    return mfccs, labels

# ========== 6. Training with Class Weights ==========
def train_with_class_weights(model, data_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, batch in enumerate(data_loader):
        if batch[0] is None:  # Skip failed batches
            continue
            
        mfccs, labels = batch
        mfccs, labels = mfccs.to(device), labels.to(device)

        # Forward pass
        outputs = model(mfccs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

# ========== 7. Enhanced Evaluation ==========
def enhanced_evaluate(model, data_loader, criterion, device, return_predictions=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            if batch[0] is None:
                continue
                
            mfccs, labels = batch
            mfccs, labels = mfccs.to(device), labels.to(device)
            outputs = model(mfccs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if return_predictions:
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    test_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    
    if return_predictions:
        return test_loss, accuracy, all_predictions, all_labels
    return test_loss, accuracy

# ========== 8. Improved Prediction Function ==========
def predict_audio_improved(model, audio_path, device, threshold=0.5):
    """Improved prediction with confidence calibration"""
    model.eval()
    with torch.no_grad():
        mfcc = extract_mfcc_robust(audio_path)
        
        # Ensure correct length
        max_len = 400
        if mfcc.shape[-1] > max_len:
            # Use center crop for prediction (more stable)
            start_idx = (mfcc.shape[-1] - max_len) // 2
            mfcc = mfcc[..., start_idx:start_idx + max_len]
        else:
            pad_amount = max_len - mfcc.shape[-1]
            mfcc = F.pad(mfcc, (0, pad_amount))
            
        mfcc = mfcc.unsqueeze(0).to(device)  # Add batch dimension
        output = model(mfcc)
        prob = torch.sigmoid(output).item()
        
        # Apply threshold
        prediction = 1 if prob > threshold else 0
        confidence = prob if prediction == 1 else 1 - prob
        language = 'Thai' if prediction == 1 else 'English'
        
        return language, confidence, prob

# ========== 9. Detect Language Function ==========
def detect_language(audio_path, model_path=None):
    """
    Detects whether the audio is in English or Thai language.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Find model file
    if model_path is None:
        model_path = "improved_final_model.pth"
        if not os.path.exists(model_path):
            model_path = "final_model.pth"
        if not os.path.exists(model_path):
            model_path = "best_model.pth"
            
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found. Please train the model first.")
    
    # Initialize and load model
    model = ImprovedLanguageDetectionCNN(dropout_rate=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Make prediction
    language, confidence, raw_prob = predict_audio_improved(model, audio_path, device)
    
    return language.lower()

# ========== 10. Main Training Script ==========
if __name__ == "__main__":
    # Set random seeds
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Training parameters
    dataset_path = "dataset"
    batch_size = 16
    num_epochs = 20
    learning_rate = 0.001
    weight_decay = 1e-4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load dataset
        dataset = LanguageDataset(dataset_path)
        
        if len(dataset) == 0:
            print("No data found! Please check your dataset path.")
            exit(1)
        
        # Split dataset
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], 
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=lambda batch: enhanced_collate_fn(batch, augment=True), 
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=lambda batch: enhanced_collate_fn(batch, augment=False), 
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=lambda batch: enhanced_collate_fn(batch, augment=False), 
            num_workers=0
        )
        
        # Calculate class weights for balanced training
        english_count = sum(1 for _, label in dataset.samples if label == 0)
        thai_count = sum(1 for _, label in dataset.samples if label == 1)
        total_samples = len(dataset.samples)
        
        # Calculate weights (inverse frequency)
        english_weight = total_samples / (2 * english_count) if english_count > 0 else 1.0
        thai_weight = total_samples / (2 * thai_count) if thai_count > 0 else 1.0
        
        print(f"Class weights - English: {english_weight:.3f}, Thai: {thai_weight:.3f}")
        
        # Initialize model and training components
        model = ImprovedLanguageDetectionCNN(dropout_rate=0.3).to(device)
        
        # Use weighted loss
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(thai_weight/english_weight).to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        counter = 0
        best_model_path = "improved_best_model.pth"
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = train_with_class_weights(
                model, train_loader, criterion, optimizer, device, epoch
            )
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
            
            # Validate
            val_loss, val_acc = enhanced_evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ Saved best model with validation loss: {val_loss:.4f}")
            else:
                counter += 1
            
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Learning rate scheduling
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"Learning rate reduced to {new_lr}")
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(best_model_path))
        test_loss, test_acc, test_preds, test_labels = enhanced_evaluate(
            model, test_loader, criterion, device, return_predictions=True
        )
        
        print(f"\nFinal Test Results:")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        
        # Detailed classification report
        test_preds = np.array(test_preds).flatten()
        test_labels = np.array(test_labels).flatten()
        
        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds, 
                                  target_names=['English', 'Thai'], 
                                  digits=3))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(test_labels, test_preds)
        print(f"              Predicted")
        print(f"              Eng  Thai")
        print(f"Actual  Eng   {cm[0,0]:3d}  {cm[0,1]:3d}")
        print(f"        Thai  {cm[1,0]:3d}  {cm[1,1]:3d}")
        
        # Save final model
        torch.save(model.state_dict(), "improved_final_model.pth")
        print("\n✅ Final model saved as 'improved_final_model.pth'")
        
        # Test prediction function
        test_path = "recorded_audio.wav"
        if os.path.exists(test_path):
            language, confidence, raw_prob = predict_audio_improved(model, test_path, device)
            print(f"\nTest prediction for {test_path}:")
            print(f"Language: {language}, Confidence: {confidence:.3f}, Raw probability: {raw_prob:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()