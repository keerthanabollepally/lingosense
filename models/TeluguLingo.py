# ============================================================
# 🔹 Multilingual Code-Mixed Translator (Telugu + English)
# ============================================================


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize import indic_normalize
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import re

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Step 1: Transliteration (Roman → Telugu)
# -------------------------------
def transliterate_roman_to_telugu(text):
    """
    Converts Roman Telugu (like 'nenu class ki vellali')
    into Telugu script ('నేను క్లాస్ కి వెల్లాలి')
    """
    try:
        telugu_text = transliterate(text, sanscript.ITRANS, sanscript.TELUGU)
        # Correct obvious mis-transliterations like చ్లస్స్ → క్లాస్
        telugu_text = telugu_text.replace("చ్లస్స్", "క్లాస్").replace("వేల్లలి", "వెళ్లాలి")
        return telugu_text
    except Exception:
        return text

# -------------------------------
# Step 2: Detect English Code-Mixed Words
# -------------------------------
def detect_code_mixed_words(text):
    return [tok for tok in indic_tokenize.trivial_tokenize(text) if re.match(r'^[a-zA-Z]+$', tok)]

# -------------------------------
# Step 3: Normalize Code-Mixed Telugu-English Text
# -------------------------------
code_mix_dict = {
    "class": "తరగతి",
    "vellali": "వెళ్లాలి",
}

def normalize_code_mixed_words(text):
    tokens = indic_tokenize.trivial_tokenize(text)
    normalizer = indic_normalize.IndicNormalizerFactory().get_normalizer("te")
    normalized_tokens = [code_mix_dict.get(tok.lower(), tok) for tok in tokens]
    normalized_text = " ".join(normalized_tokens)
    return normalizer.normalize(normalized_text)

# -------------------------------
# Step 4: Load Models
# -------------------------------
model_indic_en = "ai4bharat/indictrans2-indic-en-1B"
model_en_indic = "ai4bharat/indictrans2-en-indic-1B"

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
            use_cache=False  # <-- Set this to False to avoid error
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    translations = ip.postprocess_batch(decoded, lang=tgt_lang_tag)
    return translations[0]


# -------------------------------
# Step 6: Full Pipeline
# -------------------------------
def full_pipeline(input_text, target_langs):
    print(f"\nStep 0: Input Text: {input_text}")

    telugu_script = transliterate_roman_to_telugu(input_text)
    print(f"Step 1: After Transliteration: {telugu_script}")

    code_mix = detect_code_mixed_words(input_text)
    print(f"Step 2: Detected code-mixed words: {code_mix}")

    normalized = normalize_code_mixed_words(telugu_script)
    print(f"Step 3: Normalized Text: {normalized}")

    english_text = translate_text_safe(normalized, "tel_Telu", "eng_Latn", model_indic_en, tokenizer_indic_en)
    print(f"Step 4: English Translation: {english_text}")

    for lang in target_langs:
        translation = translate_text_safe(english_text, "eng_Latn", lang, model_en_indic, tokenizer_en_indic)
        print(f"Translation to {lang}: {translation}")

# -------------------------------
# Step 7: Run Example
# -------------------------------
input_sentence = "nenu class ki vellali"
target_languages = ["tam_Taml", "hin_Deva", "mal_Mlym", "mar_Deva", "ben_Beng"]

full_pipeline(input_sentence, target_languages)
