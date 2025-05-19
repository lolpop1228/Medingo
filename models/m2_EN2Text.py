from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
import os

class EnglishASR:
    def __init__(self):
        print("Loading English ASR model...")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
        self.model.eval()  # Set model to evaluation mode
    
    def transcribe_english(self, audio_path):
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"File not found: {audio_path}")
        
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Remove channel dimension and prepare input
        input_values = self.processor(waveform.squeeze(), return_tensors="pt", sampling_rate=sample_rate).input_values
        
        # Inference
        with torch.no_grad():
            logits = self.model(input_values=input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])
        
        return transcription

# Create a global instance of the ASR model
_asr_instance = None

def get_asr_instance():
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = EnglishASR()
    return _asr_instance

# Add this standalone function that will be imported by main_transcriber.py
def transcribe_english(audio_path):
    """
    Transcribes English audio to text.
    
    Args:
        audio_path (str): Path to the audio file to transcribe
        
    Returns:
        str: Transcribed text
    """
    asr = get_asr_instance()
    return asr.transcribe_english(audio_path)

# === Test Block ===
if __name__ == "__main__":
    audio_file = "english_audio2.wav"  # Replace with your actual .wav file path
    
    try:
        text = transcribe_english(audio_file)
        print("\nTranscription Result:")
        print(text)
    except Exception as e:
        print(f"Error: {e}")