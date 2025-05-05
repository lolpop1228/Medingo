import speech_recognition as sr
from langdetect import detect, DetectorFactory
import time
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# For reproducible language detection results
DetectorFactory.seed = 0

class ThaiEnglishDetector:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Configure recognizer for better accuracy
        self.recognizer.energy_threshold = 300  # Minimum audio energy to consider for recording
        self.recognizer.dynamic_energy_threshold = True  # Automatically adjust for ambient noise
        self.recognizer.pause_threshold = 0.8  # Seconds of non-speaking audio before a phrase is considered complete
        
        # Thai-specific linguistic features for more accurate detection
        self.thai_chars = set('กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮะัาำิีึืุูเแโใไ็่้๊๋์')
        
    def contains_thai_script(self, text):
        """Check if text contains Thai script characters (more reliable than language detection for short phrases)"""
        return any(char in self.thai_chars for char in text)
    
    def optimize_for_language(self, language):
        """Adjust recognizer settings based on detected language"""
        if language == "th":
            # Thai speech tends to be faster with different tonal qualities
            self.recognizer.pause_threshold = 0.6
        else:
            # Reset to default
            self.recognizer.pause_threshold = 0.8
            
    def determine_language(self, text):
        """Determine if text is Thai or English with higher confidence"""
        # First check for Thai script (most reliable indicator)
        if self.contains_thai_script(text):
            return "th", 0.95
        
        # For non-Thai script text, use language detection
        try:
            detected = detect(text)
            # Only accept Thai or English results
            if detected in ["th", "en"]:
                confidence = 0.8  # Base confidence
                
                # Adjust confidence based on text length
                if len(text) < 5:
                    confidence -= 0.3
                elif len(text) > 20:
                    confidence += 0.1
                    
                return detected, min(confidence, 0.95)
            else:
                # If detected as another language, check if it might be English
                # (English is often misclassified for short phrases)
                if all(ord(c) < 128 for c in text):  # ASCII characters only
                    return "en", 0.6
                else:
                    return "unknown", 0.5
        except:
            # Fallback to basic script analysis
            if all(ord(c) < 128 for c in text):  # ASCII characters only
                return "en", 0.5
            else:
                return "unknown", 0.4
    
    def listen_and_detect(self):
        """Listen for speech and detect the language"""
        with sr.Microphone() as source:
            logger.info("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            logger.info("Speak something in Thai or English...")
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                logger.info("Processing speech...")
                
                # Try with Thai language model first (helps with Thai recognition)
                try:
                    text = self.recognizer.recognize_google(audio, language="th-TH")
                    if self.contains_thai_script(text):
                        language, confidence = "th", 0.9
                    else:
                        # If no Thai script detected despite using Thai language model,
                        # retry with English model
                        raise sr.UnknownValueError("No Thai script detected")
                except sr.UnknownValueError:
                    # Try with English language model
                    text = self.recognizer.recognize_google(audio, language="en-US")
                    language, confidence = self.determine_language(text)
                
                # Print results with confidence score
                print(f"You said: {text}")
                print(f"Detected language: {'Thai' if language == 'th' else 'English'} (confidence: {confidence:.2f})")
                
                # Optimize for next detection based on current language
                self.optimize_for_language(language)
                
                return text, language, confidence
                
            except sr.WaitTimeoutError:
                logger.error("No speech detected within timeout period")
                print("No speech detected. Please try again.")
                return None, None, 0
            except sr.UnknownValueError:
                logger.error("Could not understand the audio")
                print("Sorry, I couldn't understand that. Please try again.")
                return None, None, 0
            except sr.RequestError as e:
                logger.error(f"API error: {e}")
                print("Error connecting to Google Speech Recognition service.")
                return None, None, 0
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print("An unexpected error occurred. Please try again.")
                return None, None, 0

def main():
    detector = ThaiEnglishDetector()
    
    # Allow continuous detection until user stops
    try:
        while True:
            detector.listen_and_detect()
            time.sleep(1)  # Short pause between detections
            choice = input("\nContinue listening? (y/n): ")
            if choice.lower() != 'y':
                break
    except KeyboardInterrupt:
        print("\nProgram terminated by user")

if __name__ == "__main__":
    main()