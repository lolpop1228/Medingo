from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
import os

class ThaiASR:
    def __init__(self):
        # Load pretrained Thai Wav2Vec2 model and processor from Hugging Face
        self.processor = Wav2Vec2Processor.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")
        self.model = Wav2Vec2ForCTC.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")
        self.model.eval()
    
    def transcribe_thai(self, audio_path):
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16000 Hz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Prepare input for the model
        inputs = self.processor(waveform.squeeze(), return_tensors="pt", sampling_rate=sample_rate)
        with torch.no_grad():
            logits = self.model(input_values=inputs.input_values).logits
        
        # Decode output
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])
        
        return transcription

# Create a global instance of the ASR model
_thai_asr_instance = None

def get_thai_asr_instance():
    global _thai_asr_instance
    if _thai_asr_instance is None:
        _thai_asr_instance = ThaiASR()
    return _thai_asr_instance

# Add this standalone function that will be imported by main_transcriber.py
def transcribe_thai(audio_path):
    """
    Transcribes Thai audio to text.
    
    Args:
        audio_path (str): Path to the audio file to transcribe
        
    Returns:
        str: Transcribed text
    """
    # Check if file exists
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")
        
    asr = get_thai_asr_instance()
    return asr.transcribe_thai(audio_path)

# Example usage
if __name__ == "__main__":
    try:
        audio_file = "thai_audio2.wav"  # Replace with your file path
        result = transcribe_thai(audio_file)
        print("Transcription:", result)
    except Exception as e:
        print(f"Error: {e}")