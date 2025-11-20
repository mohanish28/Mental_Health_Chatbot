# Mindful Companion - Mental Health Chatbot 🤖

**Mindful Companion** is a supportive AI chatbot designed to provide empathetic and context-aware interactions. Built with **Streamlit**, it utilizes **VADER** for sentiment analysis and **Fuzzy Matching** to understand user intents, offering a safe space for users to express their feelings.

> [!IMPORTANT]
> **Disclaimer**: This bot is for educational and supportive purposes only. It is **not** a substitute for professional mental health care. If you are in crisis, please contact emergency services immediately.

## ✨ Features

- ** empathetic Responses**: Uses sentiment analysis to gauge user emotions and tailor responses accordingly.
- **🛡️ Crisis Detection**: Automatically detects crisis keywords and provides immediate resources (e.g., helpline numbers).
- **🧠 Context Awareness**: Remembers conversation context to provide more relevant answers.
- **🗣️ Voice Interaction**: Supports voice input (via browser) and text-to-speech output for accessible communication.
- **🎨 Modern UI**: A clean, responsive interface built with Streamlit, featuring dark/light mode aesthetics and smooth animations.
- **🔍 Hybrid Matching**: Combines exact keyword matching, fuzzy logic, and sentiment analysis to find the best response.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Language**: Python 3.x
- **NLP & Logic**:
  - `nltk` (VADER Sentiment Analysis)
  - `fuzzywuzzy` (String Matching)
  - `pandas` (Data Handling)
- **Data Sources**: JSON intents, CSV FAQs, and Hugging Face datasets.

## 🚀 Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd Mental_Health_Chatbot
    ```

2.  **Install dependencies**:
    ```bash
    pip install streamlit pandas numpy fuzzywuzzy SpeechRecognition pyttsx3 nltk datasets
    ```
    *Note: You may also need to install `python-Levenshtein` for faster fuzzy matching.*

3.  **Download NLTK Data**:
    The app will automatically download the VADER lexicon on first run, but you can also do it manually:
    ```python
    import nltk
    nltk.download('vader_lexicon')
    ```

## 🏃‍♂️ Usage

Run the Streamlit application:

```bash
streamlit run models/fe.py
```

The application will open in your default web browser.

## 📂 Project Structure

```
Mental_Health_Chatbot/
├── datasets/              # Knowledge base (JSON, CSV)
├── models/
│   └── fe.py             # Main application logic
└── README.md             # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
