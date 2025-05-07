import torch
from models.m1_langDetect import LanguageDetectionCNN, extract_mfcc
from models.m2_EN2Text import EnglishASR
from models.m3_TH2Text import ThaiASR
from models.m4_EN2TH import ENtoTHTranslation
from models.m5_TH2EN import THtoENTranslation

# Initialize pre-trained models
language_detector = LanguageDetectionCNN()
language_detector.load_state_dict(torch.load("path_to_your_trained_model/language_model.pth"))  # Load the trained language detection model
language_detector.eval()  # Set the model to evaluation mode

en_asr = EnglishASR()
th_asr = ThaiASR()
en_to_th = ENtoTHTranslation()
th_to_en = THtoENTranslation()

# Function to detect language (EN or TH)
def detect_language(audio_path):
    # Step 1: Extract MFCC features from the audio
    mfcc = extract_mfcc(audio_path)
    
    # Step 2: Predict the language using the language detection model
    with torch.no_grad():
        prediction = language_detector(mfcc)  # The output will be a probability (0 or 1)
    
    # Step 3: If output is > 0.5, predict English (1), else predict Thai (0)
    predicted_language = "EN" if prediction.item() > 0.5 else "TH"
    return predicted_language

# Main function to process audio and provide translations
def process_audio(audio_path):
    # Step 1: Detect language (EN/TH)
    language = detect_language(audio_path)
    print(f"Detected language: {language}")
    
    # Step 2: Transcribe the audio and translate based on detected language
    if language == "TH":
        text = th_asr.transcribe(audio_path)
        print(f"Thai transcription: {text}")
        translated_text = th_to_en.translate(text)
        print(f"Translated to English: {translated_text}")
    else:  # For EN
        text = en_asr.transcribe(audio_path)
        print(f"English transcription: {text}")
        translated_text = en_to_th.translate(text)
        print(f"Translated to Thai: {translated_text}")
    
    return translated_text

# Example usage
audio_file_path = "path_to_audio_file.wav"
translated_text = process_audio(audio_file_path)
print(f"Final translated text: {translated_text}")
