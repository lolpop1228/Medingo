from transformers import MarianMTModel, MarianTokenizer

class ENtoTHTranslation:
    def __init__(self):
        self.model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-th")
        self.tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-th")

    def translate(self, text):
        translated = self.model.generate(**self.tokenizer(text, return_tensors="pt", padding=True))
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)

# Singleton instance
_en2th_instance = None

def translate_en_to_th(text):
    global _en2th_instance
    if _en2th_instance is None:
        _en2th_instance = ENtoTHTranslation()
    return _en2th_instance.translate(text)
