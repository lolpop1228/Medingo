import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, MarianMTModel, MarianTokenizer

# --- Language Detection Model (Simple Placeholder, replace with your own model later) ---
# This function should ideally detect if the language is EN or TH. For simplicity, we'll return 'en' here.
def detect_language(audio_path):
    # Placeholder detection (you should replace this with your actual language detection model)
    return 'en'  # In a real-world scenario, you'd return 'th' for Thai or 'en' for English

# --- EN Speech-to-Text (Wav2Vec2 for English) ---
# Load pre-trained Wav2Vec2 model and processor for English
model_en = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
processor_en = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

def transcribe_en(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    inputs = processor_en(y, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = model_en(input_values=inputs.input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor_en.decode(predicted_ids[0])
    return transcription

# --- TH Speech-to-Text (Wav2Vec2 for Thai) ---
# Load pre-trained Wav2Vec2 model and processor for Thai
model_th = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-th")
processor_th = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-th")

def transcribe_th(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    inputs = processor_th(y, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = model_th(input_values=inputs.input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor_th.decode(predicted_ids[0])
    return transcription

# --- EN-to-TH Translation (Using MarianMT) ---
model_en_to_th = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-th")
tokenizer_en_to_th = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-th")

def translate_en_to_th(text):
    inputs = tokenizer_en_to_th(text, return_tensors="pt", padding=True)
    translated = model_en_to_th.generate(**inputs)
    return tokenizer_en_to_th.decode(translated[0], skip_special_tokens=True)

# --- TH-to-EN Translation (Using MarianMT) ---
model_th_to_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-th-en")
tokenizer_th_to_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-th-en")

def translate_th_to_en(text):
    inputs = tokenizer_th_to_en(text, return_tensors="pt", padding=True)
    translated = model_th_to_en.generate(**inputs)
    return tokenizer_th_to_en.decode(translated[0], skip_special_tokens=True)

# --- Main Function to Integrate Everything ---
def main(audio_path):
    # Step 1: Detect the language (EN or TH)
    language = detect_language(audio_path)
    print(f"Detected Language: {language}")

    # Step 2: Transcribe the audio to text based on detected language
    if language == 'en':
        transcription = transcribe_en(audio_path)
        print(f"English Transcription: {transcription}")
        
        # Step 3: Optionally, translate EN to TH
        translated_th = translate_en_to_th(transcription)
        print(f"Translated to Thai: {translated_th}")
        
    elif language == 'th':
        transcription = transcribe_th(audio_path)
        print(f"Thai Transcription: {transcription}")
        
        # Step 3: Optionally, translate TH to EN
        translated_en = translate_th_to_en(transcription)
        print(f"Translated to English: {translated_en}")

if __name__ == "__main__":
    audio_path = 'path_to_your_audio.wav'  # Replace with the actual path to your audio file
    main(audio_path)
