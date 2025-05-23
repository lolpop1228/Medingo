import torch
import os

from models.m1_langDetect import LanguageDetectionCNN, extract_mfcc
from models.m2_EN2Text import EnglishASR
from models.m3_TH2Text import ThaiASR
from models.m4_EN2TH import ENtoTHTranslation
from models.m5_TH2EN import THtoENTranslation

# Load the language detection model
def load_language_detector(model_path="improved_final_model.pth"):
    model = LanguageDetectionCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Detect language from audio file
def detect_language(audio_path, model, device):
    mfcc = extract_mfcc(audio_path)
    if mfcc.dim() == 2:
        mfcc = mfcc.unsqueeze(0)  # Add batch dimension
    mfcc = mfcc.to(device)

    with torch.no_grad():
        output = model(mfcc)
        prediction = torch.sigmoid(output)

    return "EN" if prediction.item() > 0.5 else "TH"

# Process audio: detect language, transcribe, and translate
def process_audio(audio_path):
    if not os.path.exists(audio_path):
        print(f"[ERROR] File does not exist: {audio_path}")
        return None

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    language_detector = load_language_detector()
    language_detector.to(device)

    # ASR and translation models
    en_asr = EnglishASR()
    th_asr = ThaiASR()
    en_to_th = ENtoTHTranslation()
    th_to_en = THtoENTranslation()

    # Step 1: Language detection
    language = detect_language(audio_path, language_detector, device)
    print(f"Detected language: {language}")

    # Step 2: Transcribe and translate
    if language == "TH":
        text = th_asr.transcribe(audio_path)
        print(f"Thai transcription: {text}")
        translated = th_to_en.translate(text)
        print(f"Translated to English: {translated}")
    else:
        text = en_asr.transcribe(audio_path)
        print(f"English transcription: {text}")
        translated = en_to_th.translate(text)
        print(f"Translated to Thai: {translated}")

    return translated

# Entry point
if __name__ == "__main__":
    audio_path = "sample_test.wav"  # Replace with your audio file path

    translated_text = process_audio(audio_path)
    if translated_text:
        print("\nFinal Translated Output:")
        print(translated_text)
