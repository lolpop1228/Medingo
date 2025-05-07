from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

class EnglishASR:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

    def transcribe(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        inputs = self.processor(waveform, return_tensors="pt", sampling_rate=sample_rate)
        with torch.no_grad():
            logits = self.model(input_values=inputs.input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])
        return transcription
