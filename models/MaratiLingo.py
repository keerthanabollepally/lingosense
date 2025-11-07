# ------------------------------------------------------------
# Marathi Roman → Multilingual Translation Pipeline (NLLB-200)
# ------------------------------------------------------------
import re
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from indicnlp.normalize import indic_normalize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ============================================================
# 1️⃣ Load the NLLB Model and Tokenizer
# ============================================================
print("⏳ Loading NLLB model...")
model_nllb_name = "facebook/nllb-200-distilled-600M"
tokenizer_nllb = AutoTokenizer.from_pretrained(model_nllb_name)
model_nllb = AutoModelForSeq2SeqLM.from_pretrained(model_nllb_name)
print("✅ NLLB model loaded successfully!")

# ============================================================
# 2️⃣ Define translation function
# ============================================================
def translate_nllb(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate text using NLLB-200."""
    tokenizer_nllb.src_lang = src_lang
    encoded = tokenizer_nllb(text, return_tensors="pt", truncation=True, padding=True)
    generated_tokens = model_nllb.generate(
        **encoded,
        forced_bos_token_id=tokenizer_nllb.convert_tokens_to_ids(tgt_lang),
        max_length=200
    )
    return tokenizer_nllb.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# ============================================================
# 3️⃣ Roman → Marathi transliteration dictionary
# ============================================================
try:
    code_mix_dict_mr
except NameError:
    code_mix_dict_mr = {}

code_mix_dict_mr.update({
    "tu": "तू",
    "tula": "तुला",
    "kasa": "कसा",
    "kasaa": "कसा",
    "aahe": "आहे",
    "aahes": "आहेस",
    "majha": "माझा",
    "mazi": "माझी",
    "maza": "माझा",
    "bandhu": "भाऊ",
    "mitra": "मित्र",
    "aaj": "आज",
    "udya": "उद्या",
    "school": "शाळा",
    "la": "ला",
    "nako": "नको",
    "karan": "कारण",
    "majha dost": "माझा मित्र",
    "maza dost": "माझा मित्र",
    "bhet": "भेट",
    "chhan": "छान",
    "movie": "चित्रपट",
    "pahila": "पाहिला",
    "pahile": "पाहिले",
    "awesome": "छान",
    "mi": "मी",
    "jaato": "जातो",
    "jatoy": "जातो",
    "nahi": "नाही",
    "aahe ka": "आहे का",
    "dokyacha": "डोक्याचा",
    "dukh": "दुख",
    "hota": "होता",
    "mala": "मला"
})

# ============================================================
# 4️⃣ Post-transliteration cleanup
# ============================================================
_post_cleanup_map_mr = {
    "आहेस्": "आहेस",
    "कसाः": "कसा",
    "च्हान": "छान"
}

def post_transliteration_cleanup_mr(text: str) -> str:
    for bad, good in _post_cleanup_map_mr.items():
        text = text.replace(bad, good)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([अआइईउऊएऐओऔ])्", r"\1", text)
    return text

# ============================================================
# 5️⃣ Transliteration (Roman → Marathi)
# ============================================================
def transliterate_roman_to_marathi_improved(text: str) -> str:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    out_tokens = []
    for tok in tokens:
        if re.match(r"^[A-Za-z]+$", tok):
            low = tok.lower()
            if low in code_mix_dict_mr:
                out_tokens.append(code_mix_dict_mr[low])
                continue
            try:
                mar = transliterate(tok, sanscript.ITRANS, sanscript.DEVANAGARI)
                if re.search(r"[A-Za-z]", mar):
                    out_tokens.append(tok)
                else:
                    out_tokens.append(mar)
            except Exception:
                out_tokens.append(tok)
        else:
            out_tokens.append(tok)
    joined = " ".join(out_tokens).strip()
    return post_transliteration_cleanup_mr(joined)

# ============================================================
# 6️⃣ Normalize Marathi text
# ============================================================
def normalize_marathi_text_improved(text: str) -> str:
    normalizer = indic_normalize.IndicNormalizerFactory().get_normalizer("mr")
    normalized = normalizer.normalize(text)
    normalized = normalized.replace("छ्हान", "छान")
    normalized = normalized.replace("\u200c", "").replace("\u200b", "")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

# ============================================================
# 7️⃣ Telugu postprocessing (for natural tone)
# ============================================================
def postprocess_telugu(text: str) -> str:
    text = text.replace("మీరు", "నువ్వు")
    text = text.replace("మీ", "నీ")
    text = text.replace("స్నేహితుడు", "స్నేహితా")
    return re.sub(r"\s+", " ", text).strip()

# ============================================================
# 8️⃣ Helper: Prefer direct translation or fallback via English
# ============================================================
def translate_prefer_direct(src_text, src_lang, tgt_lang):
    try:
        direct = translate_nllb(src_text, src_lang, tgt_lang)
        if len(direct.strip()) > 0:
            return direct
    except Exception:
        pass
    eng = translate_nllb(src_text, src_lang, "eng_Latn")
    return translate_nllb(eng, "eng_Latn", tgt_lang)

# ============================================================
# 9️⃣ Test Pipeline
# ============================================================
def test_marathi_pipeline(examples, target_langs):
    for s in examples:
        print("\n" + ("-"*60))
        print("Input (Roman Marathi):", s)

        translit = transliterate_roman_to_marathi_improved(s)
        print("After improved transliteration:", translit)

        normalized = normalize_marathi_text_improved(translit)
        print("Normalized Marathi:", normalized)

        # English
        eng = translate_nllb(normalized, "mar_Deva", "eng_Latn")
        print("English:", eng)

        # Other Indic languages
        for lang in target_langs:
            tgt = translate_prefer_direct(normalized, "mar_Deva", lang)
            if lang == "tel_Telu":
                tgt = postprocess_telugu(tgt)
            print(f"{lang}:", tgt)
    print("\n✅ Marathi → Multilingual translation complete!")

# ============================================================
# 🔟 Run Tests
# ============================================================
examples_marathi = [
    "tu kasa aahes majha mitra",
    "mi udya school la jaato nahi karan mala dokyacha dukh hota",
    "mi movie pahila ani to khup chhan hota"
]

target_languages_mar = ["hin_Deva", "tam_Taml", "tel_Telu", "mal_Mlym", "ben_Beng"]

test_marathi_pipeline(examples_marathi, target_languages_mar)
