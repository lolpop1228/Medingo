from transformers import MarianMTModel, MarianTokenizer

class THtoENTranslation:
    def __init__(self):
        self.model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-th-en")
        self.tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-th-en")

    def translate(self, text):
        translated = self.model.generate(**self.tokenizer(text, return_tensors="pt", padding=True))
        translation = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        return translation
