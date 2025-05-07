from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa

# Load pre-trained Wav2Vec2 model and processor
model_en = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
processor_en = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

# Function to convert EN audio to text
def transcribe_en(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=16000)
    inputs = processor_en(y, return_tensors="pt", padding=True)
    
    # Perform inference
    with torch.no_grad():
        logits = model_en(input_values=inputs.input_values).logits
    
    # Decode the predicted tokens
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor_en.decode(predicted_ids[0])
    return transcription
