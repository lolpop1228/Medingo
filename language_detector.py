import speech_recognition as sr
from langdetect import detect
import logging
import time
import sys
import io

# Set up proper encoding for console output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Set up logging with utf-8 encoding
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

class ThaiEnglishDetector:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

        # Thai characters for detection
        self.thai_chars = set('กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮะัาำิีึืุูเแโใไ็่้๊๋์')

    def contains_thai_script(self, text):
        """Check if the text contains Thai script."""
        return any(char in self.thai_chars for char in text)

    def determine_language(self, text):
        """Determine whether the text is Thai or English."""
        if self.contains_thai_script(text):
            return "th", 0.95
        
        try:
            detected = detect(text)
            if detected in ["th", "en"]:
                confidence = 0.8
                if len(text) < 5:
                    confidence -= 0.3
                elif len(text) > 20:
                    confidence += 0.1
                return detected, min(confidence, 0.95)
            else:
                if all(ord(c) < 128 for c in text):
                    return "en", 0.6
                else:
                    return "unknown", 0.5
        except Exception as e:
            logger.error(f"Error during language detection: {e}")
            if all(ord(c) < 128 for c in text):
                return "en", 0.5
            else:
                return "unknown", 0.4
    
    def listen_and_detect(self):
        """Listen for speech and detect language."""
        with sr.Microphone() as source:
            logger.info("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            logger.info("Speak something in Thai or English...")
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                logger.info("Processing speech...")
                
                # Attempt to recognize speech in Thai first
                try:
                    text = self.recognizer.recognize_google(audio, language="th-TH")
                    if self.contains_thai_script(text):
                        language, confidence = "th", 0.9
                    else:
                        raise sr.UnknownValueError("No Thai script detected")
                except sr.UnknownValueError:
                    # Fallback to English
                    text = self.recognizer.recognize_google(audio, language="en-US")
                    language, confidence = self.determine_language(text)
                
                logger.info(f"Detected language: {'Thai' if language == 'th' else 'English'} (confidence: {confidence:.2f})")
                
                return language, confidence
               
            except sr.WaitTimeoutError:
                logger.error("No speech detected within timeout period")
                return None, 0
            except sr.UnknownValueError:
                logger.error("Could not understand the audio")
                return None, 0
            except sr.RequestError as e:
                logger.error(f"API error: {e}")
                return None, 0
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return None, 0


if __name__ == "__main__":
    detector = ThaiEnglishDetector()
    
    while True:
        language, confidence = detector.listen_and_detect()
        
        if language:
            print(f"Detected language: {'Thai' if language == 'th' else 'English'} (confidence: {confidence:.2f})")
        else:
            print("No speech detected or could not identify the language.")
        
        # Ask if user wants to continue listening
        choice = input("\nContinue listening? (y/n): ")
        if choice.lower() != 'y':
            break
