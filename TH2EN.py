from transformers import MarianMTModel, MarianTokenizer

model_th_to_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-th-en")
tokenizer_th_to_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-th-en")

def translate_th_to_en(text):
    inputs = tokenizer_th_to_en(text, return_tensors="pt", padding=True)
    translated = model_th_to_en.generate(**inputs)
    return tokenizer_th_to_en.decode(translated[0], skip_special_tokens=True)
