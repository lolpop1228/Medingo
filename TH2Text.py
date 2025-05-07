from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load pre-trained Wav2Vec2 model for Thai (assuming the model exists)
model_th = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-th")
processor_th = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-th")

# Function to convert TH audio to text
def transcribe_th(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    inputs = processor_th(y, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = model_th(input_values=inputs.input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor_th.decode(predicted_ids[0])
    return transcription
