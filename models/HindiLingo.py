import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indicnlp.normalize import indic_normalize
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# ----------------------------------------------------------
# DEVICE SETUP
# ----------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

# ----------------------------------------------------------
# MODELS
# ----------------------------------------------------------
MODEL_INDIC_TO_EN = "ai4bharat/indictrans2-indic-en-dist-200M"
MODEL_EN_TO_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"

print("🔄 Loading models...")
tokenizer_indic_en = AutoTokenizer.from_pretrained(MODEL_INDIC_TO_EN, trust_remote_code=True)
model_indic_en = AutoModelForSeq2SeqLM.from_pretrained(MODEL_INDIC_TO_EN, trust_remote_code=True).to(DEVICE)

tokenizer_en_indic = AutoTokenizer.from_pretrained(MODEL_EN_TO_INDIC, trust_remote_code=True)
model_en_indic = AutoModelForSeq2SeqLM.from_pretrained(MODEL_EN_TO_INDIC, trust_remote_code=True).to(DEVICE)
print("✅ Models loaded successfully!\n")

# ----------------------------------------------------------
# CODE-MIXED DICTIONARY
# ----------------------------------------------------------
code_mix_dict = {
    "class": "कक्षा", "meeting": "बैठक", "me": "में", "after": "बाद",
    "ke": "के", "baad": "बाद", "aana": "आना", "hai": "है", "mujhe": "मुझे",
    "please": "कृपया", "sir": "साहब", "madam": "मैडम"
}

# ----------------------------------------------------------
# STEP 1: ROMAN → DEVANAGARI TRANSLITERATION
# ----------------------------------------------------------
def transliterate_roman_to_hindi_deva(text: str) -> str:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    out_tokens = []
    for tok in tokens:
        if re.match(r"^[A-Za-z]+$", tok):
            mapped = code_mix_dict.get(tok.lower())
            if mapped:
                out_tokens.append(mapped)
            else:
                try:
                    devo = transliterate(tok, sanscript.ITRANS, sanscript.DEVANAGARI)
                    if re.search(r"[A-Za-z]", devo):
                        out_tokens.append(tok)
                    else:
                        out_tokens.append(devo)
                except Exception:
                    out_tokens.append(tok)
        else:
            out_tokens.append(tok)
    return " ".join(out_tokens).strip()

# ----------------------------------------------------------
# STEP 2: NORMALIZATION
# ----------------------------------------------------------
def contextualize_code_mixed_hindi(devanagari_text: str) -> str:
    normalizer = indic_normalize.IndicNormalizerFactory().get_normalizer("hi")
    normalized = normalizer.normalize(devanagari_text)
    normalized = normalized.replace("क्लास", "कक्षा").replace("मीटिंग", "बैठक")
    return normalized.strip()

# ----------------------------------------------------------
# STEP 3: CODE-MIX DETECTION
# ----------------------------------------------------------
def detect_code_mixed_words(text: str):
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return [tok for tok in tokens if tok.lower() in code_mix_dict or re.match(r"^[A-Za-z]+$", tok)]

# ----------------------------------------------------------
# STEP 4: TRANSLATION HELPERS
# ----------------------------------------------------------
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
            use_cache=False  # avoid generation cache bug
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    translations = ip.postprocess_batch(decoded, lang=tgt_lang_tag)
    return translations[0]


# ----------------------------------------------------------
# STEP 5: MULTILINGUAL TRANSLATION PIPELINE
# ----------------------------------------------------------
def full_pipeline_hindi(input_text: str, target_langs: list):
    print(f"\n🪄 Step 0: Input Text: {input_text}")

    # Hinglish → Hindi (transliteration)
    after_translit = transliterate_roman_to_hindi_deva(input_text)
    print(f"🈶 Step 1: After Transliteration (raw): {after_translit}")

    # Detect mixed tokens
    code_mix = detect_code_mixed_words(input_text)
    print(f"🔍 Step 2: Detected code-mixed words: {code_mix}")

    # Normalize
    contextual_hindi = contextualize_code_mixed_hindi(after_translit)
    print(f"🪶 Step 3: Contextual Hindi: {contextual_hindi}")

    # Hindi → English
    english_text = translate_text_safe(contextual_hindi, "hin_Deva", "eng_Latn", model_indic_en, tokenizer_indic_en)
    print(f"🇬🇧 Step 4: English Translation: {english_text}")

    # English → Other Indian languages
    print("\n🌍 Step 5: Translations to Other Languages:")
    translations = {"eng_Latn": english_text, "hin_Deva": contextual_hindi}

    for lang in target_langs:
        if lang == "hin_Deva":
            continue
        translated_text = translate_text_safe(english_text, "eng_Latn", lang, model_en_indic, tokenizer_en_indic)
        translations[lang] = translated_text
        print(f"{lang}: {translated_text}")

    print("\n✅ Translation pipeline complete!")
    return {
        "input": input_text,
        "transliterated": after_translit,
        "code_mixed": code_mix,
        "contextual_hindi": contextual_hindi,
        "english": english_text,
        "translations": translations
    }

# ----------------------------------------------------------
# EXAMPLE USAGE
# ----------------------------------------------------------
if __name__ == "__main__":
    input_sentence = "mujhe class ke baad meeting me aana hai"
    target_languages = ["tam_Taml", "tel_Telu", "mal_Mlym", "mar_Deva", "ben_Beng"]
    results = full_pipeline_hindi(input_sentence, target_languages)
