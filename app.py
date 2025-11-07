#############final(telugu , hindi, marathi, bengali ,tamil)
# =========================
# Imports
# =========================
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indicnlp.normalize import indic_normalize
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# =========================
# Device Setup
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Model Loading
# =========================
MODEL_NLLB = "facebook/nllb-200-distilled-600M"
tokenizer_nllb = AutoTokenizer.from_pretrained(MODEL_NLLB)
model_nllb = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NLLB).to(DEVICE)

# (Optional: Load IndicTrans if needed for Hindi/Tamil/Telugu and English bridging)

# =========================
# Settings (language parameters)
# =========================
LANG_CONFIG = {
    "hindi": {
        "code_mix_dict": {
            "class": "कक्षा", "meeting": "बैठक", "me": "में", "after": "बाद", "ke": "के", "baad": "बाद",
            "aana": "आना", "hai": "है", "mujhe": "मुझे", "please": "कृपया", "sir": "साहब", "madam": "मैडम"
        },
        "src_script": sanscript.ITRANS,
        "tgt_script": sanscript.DEVANAGARI,
        "lang_code_nllb": "hin_Deva",
        "normalizer": "hi"
    },
    "tamil": {
        "code_mix_dict": {
            "naan": "நான்", "nee": "நீ", "class": "கிளாஸ்", "ku": "க்கு", "poganum": "போகணும்",
            "ponanum": "போவேன்", "varanum": "வரணும்", "sollanum": "சொல்லணும்", "pananum": "பணணும்"
        },
        "src_script": sanscript.ITRANS,
        "tgt_script": sanscript.TAMIL,
        "lang_code_nllb": "tam_Taml",
        "normalizer": "ta"
    },
    "telugu": {
        "code_mix_dict": {
            "class": "తరగతి", "vellali": "వెళ్లాలి"
        },
        "src_script": sanscript.ITRANS,
        "tgt_script": sanscript.TELUGU,
        "lang_code_nllb": "tel_Telu",
        "normalizer": "te"
    },
    "marathi": {
        "code_mix_dict": {
            "tu": "तू", "tula": "तुला", "kasa": "कसा", "kasaa": "कसा", "aahe": "आहे", "aahes": "आहेस", "majha": "माझा",
            "mazi": "माझी", "maza": "माझा", "bandhu": "भाऊ", "mitra": "मित्र", "aaj": "आज", "udya": "उद्या",
            "school": "शाळा", "la": "ला", "nako": "नको", "karan": "कारण", "majha dost": "माझा मित्र",
            "maza dost": "माझा मित्र", "bhet": "भेट", "chhan": "छान", "movie": "चित्रपट",
            "pahila": "पाहिला", "pahile": "पाहिले", "awesome": "छान", "mi": "मी", "jaato": "जातो",
            "jatoy": "जातो", "nahi": "नाही", "aahe ka": "आहे का", "dokyacha": "डोक्याचा",
            "dukh": "दुख", "hota": "होता", "mala": "मला"
        },
        "src_script": sanscript.ITRANS,
        "tgt_script": sanscript.DEVANAGARI,
        "lang_code_nllb": "mar_Deva",
        "normalizer": "mr"
    },
    "bengali": {
        "code_mix_dict": {
            "ami": "আমি", "tumi": "তুমি", "tomar": "তোমার", "ke": "কে",
            "sathe": "সাথে", "bhalo": "ভালো", "jabo": "যাবো", "asche": "আসে",
            "korbo": "করবো", "amar": "আমার", "kotha": "কোথা", "bari": "বাড়ি",
            "achho": "আছো", "ki": "কি"
        },
        "src_script": sanscript.ITRANS,
        "tgt_script": sanscript.BENGALI,
        "lang_code_nllb": "ben_Beng",
        "normalizer": "bn"
    }
    
}

# =========================
# Core Functions (per language)
# =========================

def transliterate_roman_to_native(text: str, lang: str) -> str:
    config = LANG_CONFIG[lang]
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    out_tokens = []
    for tok in tokens:
        if re.match(r"^[A-Za-z]+$", tok):
            low = tok.lower()
            mapped = config["code_mix_dict"].get(low)
            if mapped:
                out_tokens.append(mapped)
            else:
                try:
                    native = transliterate(tok, config["src_script"], config["tgt_script"])
                    if re.search(r"[A-Za-z]", native):
                        out_tokens.append(tok)
                    else:
                        out_tokens.append(native)
                except Exception:
                    out_tokens.append(tok)
        else:
            out_tokens.append(tok)
    return " ".join(out_tokens).strip()

def normalize_native_text(text: str, lang: str) -> str:
    normalizer = indic_normalize.IndicNormalizerFactory().get_normalizer(LANG_CONFIG[lang]["normalizer"])
    normalized = normalizer.normalize(text)
    # Add custom replacements per language, e.g., ("क्लास", "कक्षा") for Hindi
    if lang == "hindi":
        normalized = normalized.replace("क्लास", "कक्षा").replace("मीटिंग", "बैठक")
    elif lang == "bengali":
        normalized = normalized.replace("ভাল", "ভালো")
    elif lang == "marathi":
        normalized = normalized.replace("छ्हान", "छान")
        normalized = normalized.replace("\u200c", "").replace("\u200b", "")
    return re.sub(r"\s+", " ", normalized).strip()

def detect_code_mixed_words(text: str, lang: str):
    config = LANG_CONFIG[lang]
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return [tok for tok in tokens if tok.lower() in config["code_mix_dict"] or re.match(r"^[A-Za-z]+$", tok)]

# =========================
# Translation (NLLB)
# =========================

def translate_nllb(text, src_lang, tgt_lang):
    tokenizer_nllb.src_lang = src_lang
    encoded = tokenizer_nllb(text, return_tensors="pt").to(DEVICE)
    tgt_lang_id = tokenizer_nllb.convert_tokens_to_ids(tgt_lang)
    generated = model_nllb.generate(**encoded, forced_bos_token_id=tgt_lang_id)
    return tokenizer_nllb.decode(generated[0], skip_special_tokens=True)

# =========================
# Combined Multilingual Pipeline
# =========================

def full_pipeline(input_text: str, input_lang: str, target_langs: list):
    print(f"\n🪄 Step 0: Input Text: {input_text}")

    # Step 1: Roman → Native Script
    native_text = transliterate_roman_to_native(input_text, input_lang)
    print(f"🈶 Step 1: After Transliteration: {native_text}")

    # Step 2: Detect code-mixed words
    code_mix = detect_code_mixed_words(input_text, input_lang)
    print(f"🔍 Step 2: Detected code-mixed words: {code_mix}")

    # Step 3: Normalization
    normalized_native = normalize_native_text(native_text, input_lang)
    print(f"🪶 Step 3: Normalized Text: {normalized_native}")

    # Step 4: Native → English
    english_text = translate_nllb(normalized_native, LANG_CONFIG[input_lang]["lang_code_nllb"], "eng_Latn")
    print(f"🇬🇧 Step 4: English Translation: {english_text}")

    # Step 5: English → Target Languages
    translations = {LANG_CONFIG[input_lang]["lang_code_nllb"]: normalized_native, "eng_Latn": english_text}
    print("\n🌍 Step 5: Translations to Other Languages:")
    for lang in target_langs:
        lang_code = LANG_CONFIG[lang]["lang_code_nllb"]
        translated_text = translate_nllb(english_text, "eng_Latn", lang_code)
        translations[lang_code] = translated_text
        print(f"{lang_code}: {translated_text}")

    print("\n✅ Multilingual translation pipeline complete!")
    return translations

# =========================
# Example Usage (single entry point)
# =========================
if __name__ == "__main__":
    # Set source and target languages as desired
    INPUT_SENTENCE = "mujhe class ke baad meeting me aana hai"
    INPUT_LANG = "hindi"
    TARGET_LANGUAGES = ["tamil", "telugu", "marathi", "bengali"]
    results = full_pipeline(INPUT_SENTENCE, INPUT_LANG, TARGET_LANGUAGES)
