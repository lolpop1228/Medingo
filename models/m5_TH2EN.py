from transformers import MarianMTModel, MarianTokenizer

class THtoENTranslation:
    def __init__(self):
        self.model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-th-en")
        self.tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-th-en")

    def translate(self, text):
        translated = self.model.generate(**self.tokenizer(text, return_tensors="pt", padding=True))
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)

# Singleton instance
_th2en_instance = None

def translate_th_to_en(text):
    global _th2en_instance
    if _th2en_instance is None:
        _th2en_instance = THtoENTranslation()
    return _th2en_instance.translate(text)
