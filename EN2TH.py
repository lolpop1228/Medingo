from transformers import MarianMTModel, MarianTokenizer

model_en_to_th = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-th")
tokenizer_en_to_th = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-th")

def translate_en_to_th(text):
    inputs = tokenizer_en_to_th(text, return_tensors="pt", padding=True)
    translated = model_en_to_th.generate(**inputs)
    return tokenizer_en_to_th.decode(translated[0], skip_special_tokens=True)
