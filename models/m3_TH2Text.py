from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

class ThaiASR:
    def __init__(self):
        # Load pretrained Thai Wav2Vec2 model and processor from Hugging Face
        self.processor = Wav2Vec2Processor.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")
        self.model = Wav2Vec2ForCTC.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")
        self.model.eval()

    def transcribe(self, audio_path):
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

# Example usage
if __name__ == "__main__":
    asr = ThaiASR()
    result = asr.transcribe("thai_audio2.wav")  # Replace with your file path
    print("Transcription:", result)
