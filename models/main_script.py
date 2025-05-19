import sys
from m1_langDetect import detect_language
from m2_EN2Text import transcribe_english
from m3_TH2Text import transcribe_thai

def process_audio(audio_path):
    print(f"[INFO] Processing audio file: {audio_path}")

    # Step 1: Detect language
    detected_lang = detect_language(audio_path)
    print(f"[INFO] Detected language: {detected_lang}")

    # Step 2: Route to the appropriate ASR model
    if detected_lang.lower() == 'english':
        print("[INFO] Transcribing with English ASR...")
        transcription = transcribe_english(audio_path)
    elif detected_lang.lower() == 'thai':
        print("[INFO] Transcribing with Thai ASR...")
        transcription = transcribe_thai(audio_path)
    else:
        print(f"[ERROR] Unknown language detected: {detected_lang}")
        return None

    print(f"[RESULT] Transcription:\n{transcription}")
    return transcription

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main_script.py <path_to_audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    process_audio(audio_file)
