

# ----------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------
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
print(f"💻 Using device: {DEVICE}")

# ----------------------------------------------------------
# LOAD NLLB-200 MODEL
# ----------------------------------------------------------
print("🔄 Loading NLLB-200 model (multilingual)...")
MODEL_NAME = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
print("✅ Model loaded successfully!\n")

# ----------------------------------------------------------
# CODE-MIX DICTIONARY (Roman Bengali)
# ----------------------------------------------------------
code_mix_dict_bn = {
    "ami": "আমি", "tumi": "তুমি", "tomar": "তোমার", "ke": "কে", "sathe": "সাথে",
    "bhalo": "ভালো", "jabo": "যাবো", "asche": "আসে", "korbo": "করবো",
    "amar": "আমার", "kotha": "কোথা", "bari": "বাড়ি", "achho": "আছো", "ki": "কি"
}

# ----------------------------------------------------------
# STEP 1: ROMAN → BENGALI TRANSLITERATION
# ----------------------------------------------------------
def transliterate_roman_to_bengali(text: str) -> str:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    out_tokens = []
    for tok in tokens:
        if re.match(r"^[A-Za-z]+$", tok):
            mapped = code_mix_dict_bn.get(tok.lower())
            if mapped:
                out_tokens.append(mapped)
            else:
                try:
                    bng = transliterate(tok, sanscript.ITRANS, sanscript.BENGALI)
                    if re.search(r"[A-Za-z]", bng):
                        out_tokens.append(tok)
                    else:
                        out_tokens.append(bng)
                except Exception:
                    out_tokens.append(tok)
        else:
            out_tokens.append(tok)
    return " ".join(out_tokens).strip()

# ----------------------------------------------------------
# STEP 2: NORMALIZATION
# ----------------------------------------------------------
def contextualize_code_mixed_bengali(bengali_text: str) -> str:
    normalizer = indic_normalize.IndicNormalizerFactory().get_normalizer("bn")
    normalized = normalizer.normalize(bengali_text)
    normalized = normalized.replace("ভাল", "ভালো")
    return normalized.strip()

# ----------------------------------------------------------
# STEP 3: DETECT CODE-MIXED WORDS
# ----------------------------------------------------------
def detect_code_mixed_words(text: str):
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return [tok for tok in tokens if tok.lower() in code_mix_dict_bn or re.match(r"^[A-Za-z]+$", tok)]

# ----------------------------------------------------------
# STEP 4: TRANSLATION FUNCTION (NLLB)
# ----------------------------------------------------------
def translate_nllb(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt").to(DEVICE)

    # ✅ Use convert_tokens_to_ids instead of lang_code_to_id
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if tgt_lang_id is None:
        raise ValueError(f"❌ Target language {tgt_lang} not found in tokenizer vocabulary!")

    generated = model.generate(**encoded, forced_bos_token_id=tgt_lang_id)
    return tokenizer.decode(generated[0], skip_special_tokens=True)



# ----------------------------------------------------------
# STEP 5: FULL PIPELINE
# ----------------------------------------------------------
def full_pipeline_bengali(input_text: str, target_langs: list):
    print(f"\n🪄 Step 0: Input Text: {input_text}")

    # Step 1: Roman → Bengali
    after_translit = transliterate_roman_to_bengali(input_text)
    print(f"🈶 Step 1: After Transliteration: {after_translit}")

    # Step 2: Detect code-mixed words
    code_mix = detect_code_mixed_words(input_text)
    print(f"🔍 Step 2: Detected code-mixed words: {code_mix}")

    # Step 3: Normalize
    contextual_bn = contextualize_code_mixed_bengali(after_translit)
    print(f"🪶 Step 3: Normalized Bengali: {contextual_bn}")

    # Step 4: Bengali → English
    english_text = translate_nllb(contextual_bn, "ben_Beng", "eng_Latn")
    print(f"🇬🇧 Step 4: English Translation: {english_text}")

    # Step 5: English → Other languages
    translations = {"ben_Beng": contextual_bn, "eng_Latn": english_text}
    print("\n🌍 Step 5: Translations to Other Languages:")

    for lang in target_langs:
        translated = translate_nllb(english_text, "eng_Latn", lang)
        translations[lang] = translated
        print(f"{lang}: {translated}")

    print("\n✅ Translation pipeline complete!")
    return translations

# ----------------------------------------------------------
# EXAMPLE USAGE
# ----------------------------------------------------------
if __name__ == "__main__":
    input_sentence = "ami tomar sathe jabo"  # Bengali in Roman script
    target_languages = ["hin_Deva", "tam_Taml", "tel_Telu", "mal_Mlym", "mar_Deva"]
    results = full_pipeline_bengali(input_sentence, target_languages)
