import sys
from m1_langDetect import detect_language
from m2_EN2Text import transcribe_english
from m3_TH2Text import transcribe_thai
from m4_EN2TH import translate_en_to_th
from m5_TH2EN import translate_th_to_en

def process_audio(audio_path):
    print(f"[INFO] Processing audio file: {audio_path}")

    # Step 1: Detect language
    detected_lang = detect_language(audio_path)
    print(f"[INFO] Detected language: {detected_lang}")

    # Step 2: ASR based on language
    if detected_lang.lower() == 'english':
        print("[INFO] Transcribing with English ASR...")
        transcription = transcribe_english(audio_path)
        print(f"[RESULT] English Transcription:\n{transcription}")

        print("[INFO] Translating English ➜ Thai...")
        translation = translate_en_to_th(transcription)
        print(f"[RESULT] Thai Translation:\n{translation}")

    elif detected_lang.lower() == 'thai':
        print("[INFO] Transcribing with Thai ASR...")
        transcription = transcribe_thai(audio_path)
        print(f"[RESULT] Thai Transcription:\n{transcription}")

        print("[INFO] Translating Thai ➜ English...")
        translation = translate_th_to_en(transcription)
        print(f"[RESULT] English Translation:\n{translation}")
    else:
        print(f"[ERROR] Unknown language: {detected_lang}")
        return None

    return transcription

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main_script.py <path_to_audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    process_audio(audio_file)
