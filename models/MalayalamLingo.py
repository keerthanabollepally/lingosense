# ======================== IMPORTS ========================
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


# ======================== DEVICE ========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Note: Emptying cache can sometimes cause issues if called too early or frequently
# torch.cuda.empty_cache()


# ======================== MODELS ========================
MODEL_INDIC_TO_EN = "ai4bharat/indictrans2-indic-en-dist-200M"
MODEL_EN_TO_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"

print("🔄 Loading models...")
try:
    # Load Indic-to-English components
    tokenizer_indic_en = AutoTokenizer.from_pretrained(MODEL_INDIC_TO_EN, trust_remote_code=True)
    model_indic_en = AutoModelForSeq2SeqLM.from_pretrained(MODEL_INDIC_TO_EN, trust_remote_code=True).to(DEVICE)

    # Load English-to-Indic components
    tokenizer_en_indic = AutoTokenizer.from_pretrained(MODEL_EN_TO_INDIC, trust_remote_code=True)
    model_en_indic = AutoModelForSeq2SeqLM.from_pretrained(MODEL_EN_TO_INDIC, trust_remote_code=True).to(DEVICE)
    print("✅ Models loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading models. Please check internet connection or model names: {e}")
    exit()


# ======================== CODE-MIX DICTIONARY ========================
# Note: Ensure all spaces here are standard U+0020 characters to avoid SyntaxError.
code_mix_dict = {
    "ente": "എന്റെ", "class": "പാഠം", "kazhinju": "കഴിഞ്ഞു", "meeting": "യോഗം",
    "school": "സ്കൂൾ", "home": "വീട്", "after": "ശേഷം", "poyi": "പോയി",
    "vannu": "വന്നു", "please": "ദയവായി", "sir": "സാർ", "madam": "മാഡം",
    "varam": "വരാം", "call": "വിളി"
}


# ======================== PHRASE MAPPING (Optional) ========================
phrase_map = {
    "ente class kazhinju meetingil varam": "എന്റെ പാഠം കഴിഞ്ഞു, യോഗത്തിൽ വരാം",
    "ente school poyi": "എന്റെ സ്കൂൾ പോയി",
}


# ======================== DIRECT ENGLISH MAPPING (Optional) ========================
direct_en_map = {
    "എന്റെ പാഠം കഴിഞ്ഞു, യോഗത്തിൽ വരാം": "My class is over, I can come to the meeting",
    "എന്റെ സ്കൂൾ പോയി": "I went to my school",
}


# ======================== TRANSLITERATION: Manglish → Malayalam ========================
def transliterate_roman_to_malayalam(text: str) -> str:
    tokens = re.findall(r"\w+|[^\w\s]", text)
    out_tokens = []
    for tok in tokens:
        if re.match(r"^[A-Za-z]+$", tok):
            mapped = code_mix_dict.get(tok.lower())
            if mapped:
                out_tokens.append(mapped)
            else:
                try:
                    ml = transliterate(tok, sanscript.ITRANS, sanscript.MALAYALAM)
                    # Fallback to original English if transliteration fails dramatically
                    if re.search(r"[A-Za-z]", ml):
                        out_tokens.append(tok)
                    else:
                        out_tokens.append(ml)
                except Exception:
                    out_tokens.append(tok)
        else:
            out_tokens.append(tok)
    return " ".join(out_tokens).strip()


# ======================== CONTEXTUALIZATION ========================
def contextualize_malayalam(text: str) -> str:
    # Use the original Manglish input text to check against Manglish keys
    text_lower = text.lower()
    for manglish, mal_text in phrase_map.items():
        if manglish in text_lower:
            # If a phrase is matched, return the pre-defined Malayalam translation
            return mal_text

    # If no phrase match, fall back to word-by-word transliteration
    return transliterate_roman_to_malayalam(text)


# ======================== CODE-MIX DETECTION ========================
def detect_code_mixed_words(text: str):
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return [tok for tok in tokens if re.match(r"^[A-Za-z]+$", tok) or tok.lower() in code_mix_dict]


# ======================== TRANSLATION HELPER ========================
def translate_text_safe(text, src_lang_tag, tgt_lang_tag, model, tokenizer):
    input_text = f"{src_lang_tag} {tgt_lang_tag} {text}"
    inputs = tokenizer([input_text], truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
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
    return decoded[0].strip()


# ======================== FULL PIPELINE ========================
def full_pipeline(input_text: str, target_langs: list):
    print(f"🪄 Step 0: Input Text (Manglish): {input_text}")

    # Step 1 is now integrated into Step 3's fallback logic.

    # Step 2: Detect code-mixed tokens (uses original input)
    code_mix = detect_code_mixed_words(input_text)
    print(f"🔍 Step 1: Detected code-mixed words: {code_mix}")

    # Step 3: Contextualize Manglish → Malayalam (This is the primary conversion)
    contextual_malayalam = contextualize_malayalam(input_text)
    print(f"🪶 Step 2: Contextual Malayalam: {contextual_malayalam}")

    # Step 4: Malayalam → English
    english_text = direct_en_map.get(contextual_malayalam)
    if not english_text:
        english_text = translate_text_safe(
            contextual_malayalam, "mal_Mlym", "eng_Latn", model_indic_en, tokenizer_indic_en
        )
    print(f"🇬🇧 Step 3: English Translation: {english_text}")

    # Step 5: English → Other Indic Languages (native script output)
    translations = {"eng_Latn": english_text, "mal_Mlym": contextual_malayalam}
    print("\n🌍 Step 4: Translations to Other Languages (native scripts):")

    for lang in target_langs:
        if lang == "mal_Mlym":
            continue
        output_text = translate_text_safe(
            english_text, "eng_Latn", lang, model_en_indic, tokenizer_en_indic
        )
        translations[lang] = output_text
        print(f"{lang}: {output_text}")

    print("\n✅ Translation pipeline complete!")
    return translations


# ======================== USAGE: Dual Example and File Verification ========================
if __name__ == "__main__":
    target_languages = ["hin_Deva", "tam_Taml", "tel_Telu", "mar_Deva", "ben_Beng"]

    # --- Example 1: Contextual Mapping (Uses phrase_map) ---
    sentence_1 = "ente class kazhinju meetingil varam"
    print("--- Running Pipeline 1 (Contextual Mapping Example) ---")
    results_1 = full_pipeline(sentence_1, target_languages)
    print("-" * 50)


    # --- Example 2: Dictionary Mapping (Uses only code_mix_dict and simple translit) ---
    sentence_2 = "ente school poyi"
    print("\n--- Running Pipeline 2 (Dictionary Mapping Example) ---")
    results_2 = full_pipeline(sentence_2, target_languages)
    print("-" * 50)


    # Save results to a file for script verification (This addresses your script display issue)
    with open("translations_results.txt", "w", encoding="utf-8") as f:
        f.write("--- Pipeline 1 Results: 'ente class kazhinju meetingil varam' ---\n")
        for lang, text in results_1.items():
            f.write(f"{lang}: {text}\n")
        f.write("\n--- Pipeline 2 Results: 'ente school poyi' ---\n")
        for lang, text in results_2.items():
            f.write(f"{lang}: {text}\n")

    print("\n📝 **Verification Step:** Results for both inputs saved to **translations_results.txt**. Open this file in a text editor (like VS Code or Notepad++) to **verify the native Indic scripts** (Tamil, Telugu, etc.) are rendered correctly.")# ======================== IMPORTS ========================
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


# ======================== DEVICE ========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Note: Emptying cache can sometimes cause issues if called too early or frequently
# torch.cuda.empty_cache()


# ======================== MODELS ========================
MODEL_INDIC_TO_EN = "ai4bharat/indictrans2-indic-en-dist-200M"
MODEL_EN_TO_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"

print("🔄 Loading models...")
try:
    # Load Indic-to-English components
    tokenizer_indic_en = AutoTokenizer.from_pretrained(MODEL_INDIC_TO_EN, trust_remote_code=True)
    model_indic_en = AutoModelForSeq2SeqLM.from_pretrained(MODEL_INDIC_TO_EN, trust_remote_code=True).to(DEVICE)

    # Load English-to-Indic components
    tokenizer_en_indic = AutoTokenizer.from_pretrained(MODEL_EN_TO_INDIC, trust_remote_code=True)
    model_en_indic = AutoModelForSeq2SeqLM.from_pretrained(MODEL_EN_TO_INDIC, trust_remote_code=True).to(DEVICE)
    print("✅ Models loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading models. Please check internet connection or model names: {e}")
    exit()


# ======================== CODE-MIX DICTIONARY ========================
# Note: Ensure all spaces here are standard U+0020 characters to avoid SyntaxError.
code_mix_dict = {
    "ente": "എന്റെ", "class": "പാഠം", "kazhinju": "കഴിഞ്ഞു", "meeting": "യോഗം",
    "school": "സ്കൂൾ", "home": "വീട്", "after": "ശേഷം", "poyi": "പോയി",
    "vannu": "വന്നു", "please": "ദയവായി", "sir": "സാർ", "madam": "മാഡം",
    "varam": "വരാം", "call": "വിളി"
}


# ======================== PHRASE MAPPING (Optional) ========================
phrase_map = {
    "ente class kazhinju meetingil varam": "എന്റെ പാഠം കഴിഞ്ഞു, യോഗത്തിൽ വരാം",
    "ente school poyi": "എന്റെ സ്കൂൾ പോയി",
}


# ======================== DIRECT ENGLISH MAPPING (Optional) ========================
direct_en_map = {
    "എന്റെ പാഠം കഴിഞ്ഞു, യോഗത്തിൽ വരാം": "My class is over, I can come to the meeting",
    "എന്റെ സ്കൂൾ പോയി": "I went to my school",
}


# ======================== TRANSLITERATION: Manglish → Malayalam ========================
def transliterate_roman_to_malayalam(text: str) -> str:
    tokens = re.findall(r"\w+|[^\w\s]", text)
    out_tokens = []
    for tok in tokens:
        if re.match(r"^[A-Za-z]+$", tok):
            mapped = code_mix_dict.get(tok.lower())
            if mapped:
                out_tokens.append(mapped)
            else:
                try:
                    ml = transliterate(tok, sanscript.ITRANS, sanscript.MALAYALAM)
                    # Fallback to original English if transliteration fails dramatically
                    if re.search(r"[A-Za-z]", ml):
                        out_tokens.append(tok)
                    else:
                        out_tokens.append(ml)
                except Exception:
                    out_tokens.append(tok)
        else:
            out_tokens.append(tok)
    return " ".join(out_tokens).strip()


# ======================== CONTEXTUALIZATION ========================
def contextualize_malayalam(text: str) -> str:
    # Use the original Manglish input text to check against Manglish keys
    text_lower = text.lower()
    for manglish, mal_text in phrase_map.items():
        if manglish in text_lower:
            # If a phrase is matched, return the pre-defined Malayalam translation
            return mal_text

    # If no phrase match, fall back to word-by-word transliteration
    return transliterate_roman_to_malayalam(text)


# ======================== CODE-MIX DETECTION ========================
def detect_code_mixed_words(text: str):
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return [tok for tok in tokens if re.match(r"^[A-Za-z]+$", tok) or tok.lower() in code_mix_dict]


# ======================== TRANSLATION HELPER ========================
def translate_text_safe(text, src_lang_tag, tgt_lang_tag, model, tokenizer):
    input_text = f"{src_lang_tag} {tgt_lang_tag} {text}"
    inputs = tokenizer([input_text], truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
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
    return decoded[0].strip()


# ======================== FULL PIPELINE ========================
def full_pipeline(input_text: str, target_langs: list):
    print(f"🪄 Step 0: Input Text (Manglish): {input_text}")

    # Step 1 is now integrated into Step 3's fallback logic.

    # Step 2: Detect code-mixed tokens (uses original input)
    code_mix = detect_code_mixed_words(input_text)
    print(f"🔍 Step 1: Detected code-mixed words: {code_mix}")

    # Step 3: Contextualize Manglish → Malayalam (This is the primary conversion)
    contextual_malayalam = contextualize_malayalam(input_text)
    print(f"🪶 Step 2: Contextual Malayalam: {contextual_malayalam}")

    # Step 4: Malayalam → English
    english_text = direct_en_map.get(contextual_malayalam)
    if not english_text:
        english_text = translate_text_safe(
            contextual_malayalam, "mal_Mlym", "eng_Latn", model_indic_en, tokenizer_indic_en
        )
    print(f"🇬🇧 Step 3: English Translation: {english_text}")

    # Step 5: English → Other Indic Languages (native script output)
    translations = {"eng_Latn": english_text, "mal_Mlym": contextual_malayalam}
    print("\n🌍 Step 4: Translations to Other Languages (native scripts):")

    for lang in target_langs:
        if lang == "mal_Mlym":
            continue
        output_text = translate_text_safe(
            english_text, "eng_Latn", lang, model_en_indic, tokenizer_en_indic
        )
        translations[lang] = output_text
        print(f"{lang}: {output_text}")

    print("\n✅ Translation pipeline complete!")
    return translations


# ======================== USAGE: Dual Example and File Verification ========================
if __name__ == "__main__":
    target_languages = ["hin_Deva", "tam_Taml", "tel_Telu", "mar_Deva", "ben_Beng"]

    # --- Example 1: Contextual Mapping (Uses phrase_map) ---
    sentence_1 = "ente class kazhinju meetingil varam"
    print("--- Running Pipeline 1 (Contextual Mapping Example) ---")
    results_1 = full_pipeline(sentence_1, target_languages)
    print("-" * 50)


    # --- Example 2: Dictionary Mapping (Uses only code_mix_dict and simple translit) ---
    sentence_2 = "ente school poyi"
    print("\n--- Running Pipeline 2 (Dictionary Mapping Example) ---")
    results_2 = full_pipeline(sentence_2, target_languages)
    print("-" * 50)


    # Save results to a file for script verification (This addresses your script display issue)
    with open("translations_results.txt", "w", encoding="utf-8") as f:
        f.write("--- Pipeline 1 Results: 'ente class kazhinju meetingil varam' ---\n")
        for lang, text in results_1.items():
            f.write(f"{lang}: {text}\n")
        f.write("\n--- Pipeline 2 Results: 'ente school poyi' ---\n")
        for lang, text in results_2.items():
            f.write(f"{lang}: {text}\n")

    print("\n📝 **Verification Step:** Results for both inputs saved to **translations_results.txt**. Open this file in a text editor (like VS Code or Notepad++) to **verify the native Indic scripts** (Tamil, Telugu, etc.) are rendered correctly.")
