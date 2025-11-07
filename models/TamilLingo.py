# ============================================================
# 🔹 Obligation-Aware Multilingual Roman-Tamil Translator
# ============================================================


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize import indic_normalize
import re

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Step 1: Improved Roman Tamil → Tamil Script Transliteration
# -------------------------------
def transliterate_roman_to_tamil(text):
    """
    Manual mapping for common Romanized Tamil words.
    Expand this dictionary as needed.
    """
    mapping = {
        "naan": "நான்",
        "nee": "நீ",
        "class": "கிளாஸ்",
        "ku": "க்கு",
        "poganum": "போகணும்",
        "ponanum": "போவேன்",
        "varanum": "வரணும்",
        "sollanum": "சொல்லணும்",
        "pananum": "பணணும்",
    }
    tokens = text.split()
    tamil_tokens = [mapping.get(tok.lower(), tok) for tok in tokens]
    return " ".join(tamil_tokens)

# -------------------------------
# Step 2: Detect English Code-Mixed Words
# -------------------------------
def detect_code_mixed_words(text):
    return [tok for tok in text.split() if re.match(r'^[a-zA-Z]+$', tok)]

# -------------------------------
# Step 3: Normalize Code-Mixed Tamil-English Text
# -------------------------------
code_mix_dict = {
    "class": "கிளாஸ்",
}

# Map Tamil obligation verbs to English hints
obligation_map = {
    "போகணும்": "have to go",
    "வரணும்": "have to come",
    "சொல்லணும்": "have to tell",
    "பணணும்": "have to do",
}

def normalize_code_mixed_words(text):
    tokens = text.split()
    normalizer = indic_normalize.IndicNormalizerFactory().get_normalizer("ta")
    normalized_tokens = [code_mix_dict.get(tok.lower(), tok) for tok in tokens]
    normalized_text = " ".join(normalized_tokens)
    normalized_text = normalizer.normalize(normalized_text)

    # Add English hint for obligation verbs
    for tamil_verb, eng_hint in obligation_map.items():
        if tamil_verb in normalized_text:
            normalized_text += f" ({eng_hint})"

    return normalized_text

# -------------------------------
# Step 4: Load Models
# -------------------------------
model_en_indic = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-200M")
model_indic_en = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-en-200M")

tokenizer_indic_en = AutoTokenizer.from_pretrained(model_indic_en, trust_remote_code=True)
model_indic_en = AutoModelForSeq2SeqLM.from_pretrained(model_indic_en, trust_remote_code=True).to(DEVICE)

tokenizer_en_indic = AutoTokenizer.from_pretrained(model_en_indic, trust_remote_code=True)
model_en_indic = AutoModelForSeq2SeqLM.from_pretrained(model_en_indic, trust_remote_code=True).to(DEVICE)

ip = IndicProcessor(inference=True)

# -------------------------------
# Step 5: Safe Translation Function
# -------------------------------
def translate_text_safe(text, src_lang_tag, tgt_lang_tag, model, tokenizer):
    batch = ip.preprocess_batch([text], src_lang=src_lang_tag, tgt_lang=tgt_lang_tag)
    inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=5,
            early_stopping=True,
            max_length=256,
            use_cache=False
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    translations = ip.postprocess_batch(decoded, lang=tgt_lang_tag)
    return translations[0]

# -------------------------------
# Step 6: Full Pipeline (Roman-Tamil → English → Other Languages)
# -------------------------------
def full_pipeline(input_text, target_langs):
    print(f"\nStep 0: Input Text: {input_text}")

    tamil_script = transliterate_roman_to_tamil(input_text)
    print(f"Step 1: After Transliteration: {tamil_script}")

    code_mix = detect_code_mixed_words(input_text)
    print(f"Step 2: Detected code-mixed words: {code_mix}")

    normalized = normalize_code_mixed_words(tamil_script)
    print(f"Step 3: Normalized Text: {normalized}")

    english_text = translate_text_safe(normalized, "tam_Taml", "eng_Latn", model_indic_en, tokenizer_indic_en)
    print(f"Step 4: English Translation: {english_text}")

    results = {}
    for lang in target_langs:
        translation = translate_text_safe(english_text, "eng_Latn", lang, model_en_indic, tokenizer_en_indic)
        results[lang] = translation
        print(f"Translation to {lang}: {translation}")

    return results

# -------------------------------
# Step 7: Run Example
# -------------------------------
input_sentence = "naan class ku poganum"  # Romanized Tamil
target_languages = ["tel_Telu", "hin_Deva", "mal_Mlym", "mar_Deva", "ben_Beng"]

translations = full_pipeline(input_sentence, target_languages)
