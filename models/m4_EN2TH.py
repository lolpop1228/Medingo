from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class ENtoTHTranslation:
    def __init__(self):
        model_name = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.src_lang = "eng_Latn"
        self.tgt_lang = "tha_Thai"

    def translate(self, text):
        # Set source language in tokenizer
        self.tokenizer.src_lang = self.src_lang

        # Tokenize with source language
        encoded = self.tokenizer(text, return_tensors="pt", padding=True)

        # Generate with target language ID
        generated = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
        )

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

# Singleton instance
_en2th_instance = None

def translate_en_to_th(text):
    global _en2th_instance
    if _en2th_instance is None:
        _en2th_instance = ENtoTHTranslation()
    return _en2th_instance.translate(text)
