from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class THtoENTranslation:
    def __init__(self):
        model_name = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.src_lang = "tha_Thai"
        self.tgt_lang = "eng_Latn"

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
_th2en_instance = None

def translate_th_to_en(text):
    global _th2en_instance
    if _th2en_instance is None:
        _th2en_instance = THtoENTranslation()
    return _th2en_instance.translate(text)
