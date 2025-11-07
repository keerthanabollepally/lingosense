import streamlit as st
from app import transliterate_roman_to_native, translate_nllb, LANG_CONFIG

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="LingoSense 🌐",
    layout="wide",             # 🔥 makes the app full-width (non-scrollable horizontally)
    page_icon="🌐"
)
st.markdown('<h1 class="lingosense-header">LingoSense 🌐 Indian Code-Mixed Translator</h1>', unsafe_allow_html=True)

# ===============================
# Custom CSS for Wide Layout
# ===============================
st.markdown("""
<style>
/* --- Remove the black Streamlit top header spacing --- */
div[data-testid="stHeader"] {
    height: 0rem;
    visibility: hidden;
}

/* --- Adjust main app container to move everything up --- */
.main, .block-container {
    padding-top: 0rem !important;
    margin-top: -3rem !important;  /* bring header fully into view */
}

/* --- Fix gradient header styling --- */
h1.lingosense-header {
    background: linear-gradient(90deg, #3498db, #8e44ad);
    color: white;
    font-weight: 700;
    font-size: 2rem;
    border-radius: 12px;
    padding: 28px 0;
    margin-top: 0;
    margin-bottom: 1rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    line-height: 1.3;
}

/* --- Keep background consistent --- */
.stApp {
    background-color: #0d0d14 !important;
    overflow-x: hidden;
}
</style>
""", unsafe_allow_html=True)



# Gradient header
st.markdown('<h1 class="lingosense-header">LingoSense 🌐 Indian Code-Mixed Translator</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; margin-bottom:2em">
  <span style="font-size:1.18em">Translate effortlessly across India’s diverse languages using the power of AI.<br>Powered by advanced AI.</span>
</div>
""", unsafe_allow_html=True)

# Input "card"
with st.form(key="lingosense_translate"):
    st.markdown('<div class="lingosense-card">', unsafe_allow_html=True)
    input_text = st.text_area(
        "📝 Code-mixed or Romanized sentence:", placeholder="Type or paste here...", height=64, key="main_input"
    )

    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.selectbox(
            "Source Language",
            options=list(LANG_CONFIG.keys()),
        )
    with col2:
        target_langs = st.multiselect(
            "Target Languages",
            options=["english"] + list(LANG_CONFIG.keys()),
            default=["english"]
        )

    submit = st.form_submit_button("🔄 Translate")
    st.markdown('</div>', unsafe_allow_html=True)

if submit:
    if not input_text.strip():
        st.warning("❗ Please enter a sentence.", icon="⚠️")
    else:
        st.markdown('<div class="lingosense-card">', unsafe_allow_html=True)

        st.markdown('''<span style="color:#2e70fa;font-weight:700;font-size:1em">🖋 Native Script Output</span>''', unsafe_allow_html=True)
        native_script = transliterate_roman_to_native(input_text, source_lang)
        st.code(native_script, language="")

        st.markdown('<hr class="lingosense-divider">', unsafe_allow_html=True)

        st.markdown('''<span style="color:#31b86a;font-weight:700;font-size:1em">🌍 English Translation</span>''', unsafe_allow_html=True)
        english = translate_nllb(native_script, LANG_CONFIG[source_lang]["lang_code_nllb"], "eng_Latn")
        st.success(english)

        st.markdown('<hr class="lingosense-divider">', unsafe_allow_html=True)

        st.markdown('''<span style="color:#cd5300;font-weight:700;font-size:1em">🌏 Multilingual Translations</span>''', unsafe_allow_html=True)
        for tgt in target_langs:
            if tgt == "english":
                continue
            tgt_code = LANG_CONFIG[tgt]["lang_code_nllb"]
            out = translate_nllb(english, "eng_Latn", tgt_code)
            st.markdown(f"<b style='color:#5a4cda;font-size:1.03em'>{tgt.title()}</b>", unsafe_allow_html=True)
            st.code(out, language="")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<hr>
<div style='text-align:center; color: #777; font-size: 1em; margin-top:2em;'>
  Built with ❤️ using Streamlit · <b>LingoSense</b> · 2025
</div>
""", unsafe_allow_html=True)
