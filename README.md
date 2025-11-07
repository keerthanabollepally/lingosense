# 🌍 LingoSense – Multilingual Code-Mixed Translator

**LingoSense** is an AI-powered multilingual translation tool built using **IndicTrans2** models.  
It translates **code-mixed and Romanized Indic languages** (like *Roman Telugu, Tamil, Marathi, Malayalam, Bengali, and Hindi*) into multiple Indian languages and English — all within a simple **Streamlit web app**.

---

## 🚀 Features

✅ **Supports Romanized input** — You can type *“nenu class ki vellali”* or *“naan class ku poganum”*  
✅ **Handles 6 major Indian languages:**
- Telugu  
- Tamil  
- Hindi  
- Marathi  
- Malayalam  
- Bengali  
  

✅ **Two-way Translation:**
- Indic → English  
- English → Any Indic  

✅ **Powered by AI Models:**
- `ai4bharat/indictrans2-en-indic-1B`
- `ai4bharat/indictrans2-indic-en-1B`  

✅ **Fast & Efficient** — Uses GPU (if available)  
✅ **Streamlit-based UI** — Instant results with clean interface

---

## 🧩 Project Structure

lingosense/
│
├── streamlit_app.py # Main Streamlit app
├── complete_included.py # Unified translation logic
├── requirements.txt # Dependencies
├── README.md # Documentation
└── models (each model separated)

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/LingoSense.git
cd LingoSense

### 2️⃣ Install dependencies

Make sure Python 3.9+ is installed.
Then run:

pip install -r requirements.txt

### 3️⃣ Run the app

streamlit run streamlit_app.py

Then open your browser at 👉 http://localhost:8501

🧠 How It Works
Step 1: Roman Input → Native Script

The app first transliterates Romanized text (e.g., “nenu class ki vellali”) into native script (“నేను క్లాస్ కి వెళ్లాలి”).
It uses Indic Transliteration and Indic NLP Toolkit for accurate phonetic mapping.

Step 2: Native Script → English

The IndicTrans2 Indic-to-English model (indictrans2-indic-en-1B) converts native text into English.

Step 3: English → All Indic Languages

The English-to-Indic model (indictrans2-en-indic-1B) translates the English text into all supported Indian languages.

🧰 Tech Stack

| Component            | Technology Used                          |
| -------------------- | ---------------------------------------- |
| Framework            | Streamlit                                |
| Translation Model    | IndicTrans2 (AI4Bharat)                  |
| Tokenization         | Hugging Face Transformers                |
| Transliteration      | Indic NLP Library, Indic Transliteration |
| Programming Language | Python                                   |
| Device Support       | CPU / GPU (CUDA supported)               |

🖥️ Example Usage
HINDI EXAMPLE USAGE:
<img width="1919" height="898" alt="Screenshot 2025-11-07 190514" src="https://github.com/user-attachments/assets/fa443ee1-263c-41aa-827d-432137b38b6d" />
<img width="1919" height="898" alt="Screenshot 2025-11-07 190514" src="https://github.com/user-attachments/assets/99418384-ecea-4f2a-bf53-f72bb75c9624" />

MARATHI EXAMPLE USAGE:
<img width="1868" height="901" alt="Screenshot 2025-11-07 190417" src="https://github.com/user-attachments/assets/f89e8d3f-b6fd-4c27-877c-d392a57f0f5f" />
<img width="1879" height="742" alt="Screenshot 2025-11-07 190429" src="https://github.com/user-attachments/assets/58cc153a-8978-47a9-8e45-9b4ab4b5c9c3" />


Input (Roman Telugu):

nenu class ki vellali

Output:

| Language  | Translation                     |
| --------- | ------------------------------- |
| English   | I need to go to class           |
| Hindi     | मुझे कक्षा में जाना है          |
| Tamil     | எனக்கு வகுப்புக்கு போக வேண்டும் |
| Malayalam | എനിക്ക് ക്ലാസിലേക്കു പോകണം      |
| Bengali   | আমাকে ক্লাসে যেতে হবে           |
| Marathi   | मला वर्गात जायचे आहे            |
| Telugu    | నేను క్లాస్ కి వెళ్లాలి         |

🧩 Add More Languages:

Add more Indic languages by extending the target_langs list:
target_langs = ["hin_Deva", "tam_Taml", "mal_Mlym", "mar_Deva", "ben_Beng", "tel_Telu", "kan_Knda"]

💡 Future Enhancements

- 🔊 Voice-to-Text translation (via Whisper)

- 🗣️ Speech output in regional languages

- 💬 Chat-based multilingual assistant

- 🧠 Offline translation with model quantization


