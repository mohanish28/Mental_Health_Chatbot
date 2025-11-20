# app.py
import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import process, fuzz
from speech_recognition import Recognizer, Microphone, UnknownValueError, RequestError
import pyttsx3
import time
import json
import os
from datasets import load_dataset
import random
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from spellchecker import SpellChecker

# Download VADER lexicon if not present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Set page config (must be first Streamlit command)
st.set_page_config(
    page_title="Mindful Companion",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, attractive UI
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 120px; /* Space for fixed input */
    }
    
    /* Chat container - scrollable area */
    .chat-container {
        max-height: calc(100vh - 280px);
        overflow-y: auto;
        overflow-x: hidden;
        padding: 1rem;
        margin-bottom: 1rem;
        scroll-behavior: smooth;
        scrollbar-width: thin;
        scrollbar-color: #667eea #f1f1f1;
    }
    
    .chat-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Fixed input area at bottom - blend with background */
    .fixed-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: transparent;
        padding: 1rem 1.5rem;
        z-index: 999;
    }
    
    /* Style form within fixed container */
    .fixed-input-container .stForm {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 1rem 1rem 0 0;
        padding: 1rem;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.08);
        border-top: 1px solid rgba(226, 232, 240, 0.6);
    }
    
    .fixed-input-container .stTextInput > div > div > input {
        background: white;
    }
    
    /* Adjust main content for fixed input */
    .main .block-container {
        padding-bottom: 140px !important;
    }
    
    /* Adjust for sidebar on larger screens */
    @media (min-width: 768px) {
        .fixed-input-container {
            left: 21rem;
        }
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .fixed-input-container {
            left: 0;
            padding: 0.75rem;
        }
        .chat-container {
            max-height: calc(100vh - 200px);
        }
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
    }
    
    h3 {
        color: #667eea !important;
        font-weight: 600 !important;
    }
    
    /* Chat message animations */
    @keyframes fadeIn {
        from { 
            opacity: 0; 
            transform: translateY(20px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .chat-message {
        padding: 1.2rem 1.5rem;
        border-radius: 1.5rem;
        margin: 1rem 0;
        animation: fadeIn 0.4s ease-out;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        max-width: 75%;
        word-wrap: break-word;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        margin-right: 0;
        border-bottom-right-radius: 0.3rem;
        animation: slideIn 0.3s ease-out;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #2d3748;
        margin-right: auto;
        margin-left: 0;
        border-bottom-left-radius: 0.3rem;
        animation: slideIn 0.3s ease-out;
    }
    
    /* Status indicator */
    .status-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Typing animation */
    .typing-animation {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 1rem;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #667eea;
        border-radius: 50%;
        display: inline-block;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 100% { 
            transform: translateY(0);
            opacity: 0.7;
        }
        50% { 
            transform: translateY(-10px);
            opacity: 1;
        }
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1.25rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
    }
    
    /* Info box styling */
    .stInfo {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        border-left: 4px solid #667eea;
        border-radius: 0.5rem;
    }
    
    /* Hide loading messages in production */
    .stSuccess, .stWarning, .stError, .stInfo {
        display: none;
    }

    /* Suggested topics styling */
    .suggestion-chip {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.2rem;
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        color: #667eea;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .suggestion-chip:hover {
        background: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour in production
def load_datasets():
    """Load and combine multiple datasets with context support"""
    combined_dataset = {}
    
    try:
        # 1. Load local JSON dataset (intents.json) - High priority
        json_paths = [
            "datasets/intents.json",
            os.path.join(os.path.dirname(__file__), "..", "datasets", "intents.json"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "intents.json")
        ]
        
        json_dataset = {}
        for json_path in json_paths:
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        json_data = json.load(f)
                        for intent in json_data.get("intents", []):
                            for pattern in intent.get("patterns", []):
                                pattern_lower = pattern.lower().strip()
                                if pattern_lower:
                                    # Store full intent data
                                    json_dataset[pattern_lower] = {
                                        "response": random.choice(intent.get("responses", ["I'm here to help."])),
                                        "context_set": intent.get("context_set", ""),
                                        "context_filter": intent.get("context_filter", ""),
                                        "tag": intent.get("tag", "")
                                    }
                    break
                except Exception:
                    pass
        
        # 2. Load local CSV dataset (Mental_Health_FAQ.csv) - High priority
        csv_paths = [
            "datasets/Mental_Health_FAQ.csv",
            os.path.join(os.path.dirname(__file__), "..", "datasets", "Mental_Health_FAQ.csv"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "Mental_Health_FAQ.csv")
        ]
        
        csv_dataset = {}
        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path, encoding="utf-8")
                    df.columns = df.columns.str.strip()
                    
                    if "Context" in df.columns and "Response" in df.columns:
                        for _, row in df.iterrows():
                            context = str(row["Context"]).lower().strip()
                            response = str(row["Response"]).strip()
                            if context and response:
                                csv_dataset[context] = {
                                    "response": response,
                                    "context_set": "",
                                    "context_filter": ""
                                }
                    break
                except Exception:
                    pass
        
        # 3. Load Hugging Face dataset - Additional data source (not just fallback)
        hf_dataset = {}
        try:
            ds = load_dataset("Amod/mental_health_counseling_conversations", trust_remote_code=True)
            df = pd.DataFrame(ds["train"])
            # Sample a subset for performance (first 1000 entries)
            df_sample = df.head(1000) if len(df) > 1000 else df
            for _, row in df_sample.iterrows():
                context = str(row.get("Context", "")).lower().strip()
                response = str(row.get("Response", "")).strip()
                if context and response and len(context) > 3:
                    # Only add if not already in local datasets (local takes priority)
                    if context not in json_dataset and context not in csv_dataset:
                        hf_dataset[context] = {
                            "response": response,
                            "context_set": "",
                            "context_filter": ""
                        }
        except Exception:
            pass
        
        # Combine all datasets: JSON (highest priority) > CSV > Hugging Face
        combined_dataset = {**hf_dataset, **csv_dataset, **json_dataset}
        
        return combined_dataset
        
    except Exception:
        return {}

def get_sentiment(text):
    """Calculate sentiment score using VADER"""
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']

def correct_sentence(text):
    """Correct misspelled words in a sentence"""
    spell = SpellChecker()
    words = text.split()
    corrected_words = []
    
    for word in words:
        # Get the one `most likely` answer
        corrected = spell.correction(word)
        # If correction returns None (rare), keep original
        corrected_words.append(corrected if corrected else word)
        
    return " ".join(corrected_words)

def get_best_response(user_input, dataset):
    """Advanced matching with personalization, sentiment analysis, and context awareness"""
    if not user_input:
        return "I'm here to listen. Please tell me how you're feeling."
    
    if not dataset:
        return "I'm here to support you. Tell me more."
    
    # Personalization: Get user name
    user_name = st.session_state.get("user_name", "")
    name_prefix = f"{user_name}, " if user_name else ""
    
    # Context Management
    if "context" not in st.session_state:
        st.session_state.context = ""
    
    current_context = st.session_state.context
    
    # Sentiment Analysis
    sentiment_score = get_sentiment(user_input)
    
    # Context Tracking (Simple)
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(user_input)
    
    # Spell Correction
    user_input_corrected = correct_sentence(user_input)
    if user_input_corrected != user_input:
        # Optional: Show corrected text for transparency (or just use it silently)
        # st.caption(f"Did you mean: {user_input_corrected}?")
        pass
        
    user_input_lower = user_input_corrected.lower().strip()
    user_words = [w for w in user_input_lower.split() if len(w) > 1]
    user_words_set = set(user_words)
    
    # Step 0: Critical Crisis Detection (Highest Priority)
    crisis_keywords = ["suicide", "kill myself", "want to die", "end my life", "hurt myself"]
    if any(keyword in user_input_lower for keyword in crisis_keywords):
        return "I'm very sorry you're feeling this way, but I want you to be safe. Please reach out to a crisis hotline immediately. You can call 988 (in the US) or text HOME to 741741. You are not alone, and there is help available."

    # Helper to process match
    def process_match(match_data):
        # Check context filter
        if match_data.get("context_filter") and match_data["context_filter"] != current_context:
            return None
            
        # Update context if set
        if match_data.get("context_set"):
            st.session_state.context = match_data["context_set"]
            
        response = match_data["response"]
        
        # Handle {name} placeholder
        if "{name}" in response:
            if user_name:
                response = response.replace("{name}", user_name)
            else:
                response = response.replace(" {name}", "").replace("{name}", "")
        elif user_name and random.random() < 0.3:
            response = f"{name_prefix}{response.lower() if response[0].islower() else response}"
            
        return response

    # Step 1: Exact match (highest priority)
    if user_input_lower in dataset:
        result = process_match(dataset[user_input_lower])
        if result: return result
    
    # Step 2: Special handling for time-based greetings
    time_greetings = ["good morning", "good afternoon", "good evening", "good night"]
    for greeting in time_greetings:
        if greeting in user_input_lower and greeting in dataset:
            result = process_match(dataset[greeting])
            if result: return result
    
    # Step 3: Advanced phrase matching
    best_matches = []
    
    for key, data in dataset.items():
        # Optimization: Skip if context filter doesn't match current context
        if data.get("context_filter") and data["context_filter"] != current_context:
            continue
            
        key_lower = key.lower()
        key_words = [w for w in key_lower.split() if len(w) > 1]
        
        if not key_words: continue
        
        # Skip time greetings logic (same as before)
        if user_input_lower == "good" and any(g in key_lower for g in time_greetings): continue
        if any(g in key_lower for g in time_greetings) and not any(g in user_input_lower for g in time_greetings): continue
        
        # Calculate score
        key_words_set = set(key_words)
        common_words = user_words_set.intersection(key_words_set)
        
        word_overlap = len(common_words) / max(len(user_words), len(key_words), 1)
        length_similarity = 1 - abs(len(user_words) - len(key_words)) / max(len(user_words), len(key_words), 1)
        
        order_score = 0
        if len(user_words) > 1 and len(key_words) > 1:
            user_bigrams = set(zip(user_words[:-1], user_words[1:]))
            key_bigrams = set(zip(key_words[:-1], key_words[1:]))
            if user_bigrams and key_bigrams:
                order_score = len(user_bigrams.intersection(key_bigrams)) / max(len(user_bigrams), len(key_bigrams), 1)
        
        substring_score = 1.0 if key_lower in user_input_lower or user_input_lower in key_lower else 0.0
        
        combined_score = (word_overlap * 0.4 + length_similarity * 0.2 + order_score * 0.2 + substring_score * 0.2)
        
        if len(key_words) == 1 and combined_score >= 0.9:
            best_matches.append((data, combined_score))
        elif len(key_words) == 2 and len(common_words) == 2 and combined_score >= 0.85:
            best_matches.append((data, combined_score))
        elif len(common_words) >= 2 and combined_score >= 0.75:
            best_matches.append((data, combined_score))
    
    if best_matches:
        best_matches.sort(key=lambda x: x[1], reverse=True)
        return process_match(best_matches[0][0])
    
    # Step 4: SVM Prediction (New High Priority)
    try:
        # Load model if not already loaded
        if "svm_model" not in st.session_state:
            try:
                import pickle
                model_path = os.path.join(os.path.dirname(__file__), "svm_model.pkl")
                with open(model_path, "rb") as f:
                    st.session_state.svm_model = pickle.load(f)
            except Exception:
                st.session_state.svm_model = None

        if st.session_state.svm_model:
            # Predict
            probas = st.session_state.svm_model.predict_proba([user_input_lower])[0]
            max_proba = max(probas)
            predicted_tag = st.session_state.svm_model.classes_[probas.argmax()]
            
            # Threshold for SVM confidence
            if max_proba >= 0.5:  # Lowered threshold slightly as SVM probabilities can be conservative
                # Find a response for this tag
                # We search the dataset for any entry with this tag
                possible_responses = []
                for data in dataset.values():
                    if data.get("tag") == predicted_tag:
                        # Check context filter
                        if data.get("context_filter") and data["context_filter"] != current_context:
                            continue
                        possible_responses.append(data)
                
                if possible_responses:
                    # Pick a random one from the matching tag group to add variety
                    chosen_data = random.choice(possible_responses)
                    result = process_match(chosen_data)
                    if result: return result

    except Exception as e:
        pass
        
    # Step 5: Fuzzy matching with strict threshold
    try:
        # Filter time greetings if needed
        keys_to_search = list(dataset.keys())
        if user_input_lower == "good":
            keys_to_search = [k for k in keys_to_search if not any(g in k.lower() for g in time_greetings)]
        
        if keys_to_search:
            best_match, fuzzy_score = process.extractOne(user_input_lower, keys_to_search, scorer=fuzz.token_sort_ratio)
            if fuzzy_score >= 80:  # High threshold for fuzzy matches
                result = process_match(dataset[best_match])
                if result: return result
    except Exception:
        pass
    
    # Step 5: Context-aware fallback responses
    # Analyze user input sentiment/keywords for better fallback
    positive_words = ["good", "great", "fine", "okay", "ok", "happy", "well", "better"]
    negative_words = ["bad", "sad", "depressed", "anxious", "stressed", "worried", "scared", "lonely"]
    question_words = ["what", "how", "why", "when", "where", "who", "which"]
    
    user_lower = user_input_lower
    has_positive = any(word in user_lower for word in positive_words)
    has_negative = any(word in user_lower for word in negative_words)
    is_question = any(word in user_lower for word in question_words) or user_lower.endswith("?")
    
    # Contextual fallback responses
    # Contextual fallback responses based on sentiment
    if sentiment_score <= -0.5: # Very negative
        return random.choice([
            f"I can hear how difficult this is for you{', ' + user_name if user_name else ''}. I'm here to listen.",
            f"It sounds like you're going through a lot right now{', ' + user_name if user_name else ''}. I'm here for you.",
            "I'm so sorry you're feeling this way. Your feelings are valid."
        ])
    elif sentiment_score >= 0.5: # Very positive
        return random.choice([
            f"It's wonderful to hear that positivity{', ' + user_name if user_name else ''}!",
            "I'm really glad to hear that! Keep going.",
            f"That sounds great{', ' + user_name if user_name else ''}!"
        ])
    elif has_positive:
        return random.choice([
            "That's wonderful to hear!",
            "I'm glad you're feeling good.",
            "That's great!"
        ])
    elif has_negative:
        return random.choice([
            "I'm sorry to hear that.",
            "It sounds like you're going through a difficult time. I'm here.",
            "I understand this is hard."
        ])
    elif is_question:
        return random.choice([
            "That's a good question. Can you say more?",
            "I'd be happy to help. Could you clarify?",
            "Let's explore that."
        ])
    else:
        return random.choice([
            f"I'm here{', ' + user_name if user_name else ''}.",
            "I understand.",
            "Thank you for sharing.",
            "I'm listening.",
            "Go on."
        ])

def recognize_speech():
    """Speech to text conversion - browser-based (no PyAudio needed)"""
    # This function is kept for compatibility but actual recognition happens in browser
    # The browser-based implementation is in the JavaScript code
    return ""

def text_to_speech(text):
    """Text to speech conversion"""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    engine.say(text)
    engine.runAndWait()

def chat_bubble(message, is_user=False):
    """Create animated chat bubble"""
    cls = "user-message" if is_user else "bot-message"
    st.markdown(f"""
    <div class="chat-message {cls}">
        {message}
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application"""
    # Initialize session state (cached dataset loading)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.last_processed = None  # Track last processed input
    
    # Load dataset (cached for performance)
    if "dataset" not in st.session_state:
        with st.spinner("Initializing... Please wait."):
            st.session_state.dataset = load_datasets()
    
    # Compact header with gradient
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
        <h1 style="margin-bottom: 0.3rem; font-size: 2.5rem;">🤖 Mindful Companion</h1>
        <p style="font-size: 1rem; color: #64748b; margin-top: 0;">Your 24/7 Mental Health Support</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with helpful information
    with st.sidebar:
        st.markdown("### 👤 Profile")
        if "user_name" not in st.session_state:
            st.session_state.user_name = ""
        
        name_input = st.text_input("Your Name (Optional)", value=st.session_state.user_name, placeholder="Enter your name...")
        if name_input != st.session_state.user_name:
            st.session_state.user_name = name_input
            st.rerun()
            
        st.markdown("---")
        st.markdown("### 💚 Status")
        st.markdown("<div class='status-indicator' style='background: #10b981;'></div> <span style='color: #10b981; font-weight: 600;'>System Active</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); 
                    padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #667eea;">
            <p style="margin: 0.5rem 0;">💬 Type your message and press Enter</p>
            <p style="margin: 0.5rem 0;">💭 Speak naturally like you're talking to a friend</p>
            <p style="margin: 0.5rem 0;">🎤 Voice input is optional (if available)</p>
            <p style="margin: 0.5rem 0;">❌ Type 'exit' to end conversation</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### ⚠️ Important")
        st.markdown("""
        <div style="background: #fef3c7; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b;">
            <p style="margin: 0; font-size: 0.9rem; color: #92400e;">
                This is not a substitute for professional mental health care. 
                If you're in crisis, please contact emergency services.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Chat messages container (scrollable)
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="chat-container" id="chatContainer">', unsafe_allow_html=True)
        
        # Display chat messages
        if st.session_state.messages:
            for message in st.session_state.messages:
                chat_bubble(message["content"], is_user=message["role"] == "user")
        else:
            # Welcome message
            welcome_msg = f"Hello{', ' + st.session_state.user_name if st.session_state.user_name else ''}! I'm here to listen and support you. How are you feeling today?"
            chat_bubble(welcome_msg, is_user=False)
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Auto-scroll to bottom script
        st.markdown("""
        <script>
            function scrollToBottom() {
                const container = document.querySelector('.chat-container');
                if (container) {
                    container.scrollTop = container.scrollHeight;
                }
            }
            // Scroll on load and after content updates
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', scrollToBottom);
            } else {
                scrollToBottom();
            }
            // Scroll after render
            setTimeout(scrollToBottom, 200);
            // Use MutationObserver to auto-scroll when new messages are added
            const observer = new MutationObserver(scrollToBottom);
            const chatContainer = document.querySelector('.chat-container');
            if (chatContainer) {
                observer.observe(chatContainer, { childList: true, subtree: true });
            }
        </script>
        """, unsafe_allow_html=True)

    # Suggested topics area (above input)
    if not st.session_state.messages:
        st.markdown('<div style="margin-left: 1rem; margin-bottom: 0.5rem; color: #64748b; font-size: 0.9rem;">Try asking about:</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        topics = ["Panic Attack", "Anxiety Tips", "Sleep Help", "Grounding"]
        
        # Function to handle topic click
        def set_topic(t):
            st.session_state.input_value = t
        
        for i, topic in enumerate(topics):
            with cols[i]:
                if st.button(topic, key=f"topic_{i}", use_container_width=True):
                    # We can't easily set the input value directly without a rerun and session state trickery
                    # So we'll just append it to messages directly to simulate sending
                    st.session_state.messages.append({"role": "user", "content": topic})
                    response = get_best_response(topic, st.session_state.dataset)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
    
    # Fixed input area at bottom
    st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)
    
    # Use a form to handle input properly and avoid rerun loops
    with st.form(key="chat_form", clear_on_submit=True):
        # Main text input (primary method)
        user_input = st.text_input(
            "Type your message...", 
            key="input",
            label_visibility="collapsed",
            placeholder="💬 Type your message here and press Enter to send..."
        )
        
        # Form submit button row
        col1, col2 = st.columns([10, 1])
        with col1:
            submitted = st.form_submit_button("Send 💬", use_container_width=True, type="primary")
        with col2:
            voice_button = st.form_submit_button("🎤", use_container_width=True, help="Click to start voice input")
    
    # Initialize voice recognition state
    if "voice_listening" not in st.session_state:
        st.session_state.voice_listening = False
    if "voice_text" not in st.session_state:
        st.session_state.voice_text = ""
    
    # Add browser-based speech recognition JavaScript
    st.markdown("""
    <script>
        (function() {
            let recognition = null;
            let isListening = false;
            
            // Initialize Web Speech API
            function initRecognition() {
                if (!recognition && ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
                    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                    recognition = new SpeechRecognition();
                    recognition.continuous = false;
                    recognition.interimResults = false;
                    recognition.lang = 'en-US';
                    
                    recognition.onstart = function() {
                        isListening = true;
                        showVoiceIndicator('🎤 Listening...');
                        console.log('Voice recognition started');
                    };
                    
                    recognition.onresult = function(event) {
                        const transcript = event.results[0][0].transcript;
                        isListening = false;
                        showVoiceIndicator('✅ Captured: ' + transcript);
                        console.log('Voice captured:', transcript);
                        
                        // Fill input and submit form
                        setTimeout(() => {
                            const input = document.querySelector('input[type="text"]');
                            if (input && input.placeholder && input.placeholder.includes('Type your message')) {
                                // Set the value properly for Streamlit
                                input.value = transcript;
                                
                                // Trigger input events for Streamlit
                                const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
                                nativeInputValueSetter.call(input, transcript);
                                
                                const inputEvent = new Event('input', { bubbles: true });
                                input.dispatchEvent(inputEvent);
                                
                                const changeEvent = new Event('change', { bubbles: true });
                                input.dispatchEvent(changeEvent);
                                
                                // Submit the form after a short delay
                                setTimeout(() => {
                                    const form = input.closest('form');
                                    if (form) {
                                        // Find the Send button (not the voice button)
                                        const buttons = form.querySelectorAll('button[type="submit"]');
                                        if (buttons.length > 0) {
                                            // Click the first submit button (Send button)
                                            buttons[0].click();
                                        }
                                    }
                                }, 300);
                            }
                            hideVoiceIndicator();
                        }, 800);
                    };
                    
                    recognition.onerror = function(event) {
                        isListening = false;
                        console.error('Recognition error:', event.error);
                        if (event.error === 'no-speech') {
                            showVoiceIndicator('❌ No speech detected. Try again.');
                        } else if (event.error === 'not-allowed') {
                            showVoiceIndicator('❌ Microphone permission denied. Please allow access.');
                        } else {
                            showVoiceIndicator('❌ Error: ' + event.error);
                        }
                        setTimeout(hideVoiceIndicator, 2000);
                    };
                    
                    recognition.onend = function() {
                        isListening = false;
                        console.log('Voice recognition ended');
                    };
                    
                    // Make recognition globally available
                    window.voiceRecognition = recognition;
                    window.isVoiceListening = () => isListening;
                    window.startVoiceRecognition = function() {
                        console.log('startVoiceRecognition called, recognition:', recognition, 'isListening:', isListening);
                        if (!recognition) {
                            initRecognition();
                        }
                        if (recognition) {
                            if (isListening) {
                                console.log('Stopping recognition');
                                recognition.stop();
                            } else {
                                try {
                                    console.log('Starting recognition...');
                                    recognition.start();
                                } catch (err) {
                                    console.error('Error starting recognition:', err);
                                    if (err.message && err.message.includes('already started')) {
                                        recognition.stop();
                                        setTimeout(() => recognition.start(), 100);
                                    } else {
                                        alert('Could not start voice recognition: ' + err.message);
                                    }
                                }
                            }
                        } else {
                            alert('Speech recognition is not supported in your browser.\\n\\nPlease use:\\n• Google Chrome\\n• Microsoft Edge\\n• Safari\\n\\nFirefox does not support this feature.');
                        }
                    };
                } else {
                    console.warn('Speech recognition not available in this browser');
                }
            }
            
            function showVoiceIndicator(message) {
                let indicator = document.getElementById('voice-status-indicator');
                if (!indicator) {
                    indicator = document.createElement('div');
                    indicator.id = 'voice-status-indicator';
                    indicator.style.cssText = 'position: fixed; bottom: 100px; left: 50%; transform: translateX(-50%); background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem 2rem; border-radius: 30px; z-index: 1001; font-weight: 600; box-shadow: 0 6px 20px rgba(0,0,0,0.3); font-size: 1rem; white-space: nowrap;';
                    document.body.appendChild(indicator);
                }
                indicator.textContent = message;
                indicator.style.display = 'block';
            }
            
            function hideVoiceIndicator() {
                const indicator = document.getElementById('voice-status-indicator');
                if (indicator) {
                    indicator.style.display = 'none';
                }
            }
            
            // Initialize recognition on load
            initRecognition();
            
            // Handle voice button clicks - more aggressive interception
            function setupVoiceButtonHandler() {
                // Find all submit buttons
                const buttons = document.querySelectorAll('button[type="submit"]');
                console.log('Found buttons:', buttons.length);
                
                buttons.forEach((button, index) => {
                    const buttonText = (button.textContent || button.innerText || '').trim();
                    console.log('Button', index, 'text:', buttonText);
                    
                    // Check if this is the voice button (contains microphone emoji)
                    if (buttonText.includes('🎤')) {
                        console.log('Found voice button!');
                        
                        // Remove any existing handlers by cloning
                        if (!button.hasAttribute('data-voice-setup')) {
                            button.setAttribute('data-voice-setup', 'true');
                            
                            // Add multiple event listeners to catch the click
                            ['click', 'mousedown', 'touchstart'].forEach(eventType => {
                                button.addEventListener(eventType, function(e) {
                                    console.log('Voice button clicked!', eventType);
                                    
                                    e.preventDefault();
                                    e.stopPropagation();
                                    e.stopImmediatePropagation();
                                    
                                    // Prevent form submission
                                    const form = button.closest('form');
                                    if (form) {
                                        const preventSubmit = function(ev) {
                                            ev.preventDefault();
                                            ev.stopPropagation();
                                            ev.stopImmediatePropagation();
                                            return false;
                                        };
                                        form.addEventListener('submit', preventSubmit, { capture: true, once: true });
                                    }
                                    
                                    // Start voice recognition immediately
                                    if (window.startVoiceRecognition) {
                                        window.startVoiceRecognition();
                                    } else {
                                        initRecognition();
                                        setTimeout(() => {
                                            if (window.startVoiceRecognition) {
                                                window.startVoiceRecognition();
                                            } else {
                                                alert('Voice recognition failed to initialize. Please refresh the page.');
                                            }
                                        }, 100);
                                    }
                                    
                                    return false;
                                }, true); // Capture phase
                            });
                        }
                    }
                });
            }
            
            // Setup handlers immediately and on DOM ready
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', function() {
                    setupVoiceButtonHandler();
                    // Also setup after a delay to catch Streamlit's dynamic rendering
                    setTimeout(setupVoiceButtonHandler, 300);
                    setTimeout(setupVoiceButtonHandler, 1000);
                });
            } else {
                setupVoiceButtonHandler();
                setTimeout(setupVoiceButtonHandler, 300);
                setTimeout(setupVoiceButtonHandler, 1000);
            }
            
            // Re-setup handlers after Streamlit reruns (using MutationObserver)
            const observer = new MutationObserver(function(mutations) {
                setupVoiceButtonHandler();
            });
            observer.observe(document.body, { childList: true, subtree: true });
        })();
    </script>
    """, unsafe_allow_html=True)
    
    # Handle voice button click from Python side - trigger recognition
    if voice_button:
        st.markdown("""
        <script>
            console.log('Voice button clicked from Python side');
            // Try multiple times to ensure it works
            function tryStartRecognition(attempts) {
                if (window.startVoiceRecognition) {
                    console.log('Calling startVoiceRecognition, attempt:', attempts);
                    window.startVoiceRecognition();
                } else if (attempts < 5) {
                    console.log('startVoiceRecognition not ready, retrying...', attempts);
                    setTimeout(() => tryStartRecognition(attempts + 1), 200);
                } else {
                    console.error('Failed to start recognition after multiple attempts');
                    alert('Voice recognition is not ready. Please try clicking the button again.');
                }
            }
            setTimeout(() => tryStartRecognition(1), 100);
        </script>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process user input (primary method - text input)
    # Only process if form was submitted and it's a new message
    if submitted and user_input:
        user_input_clean = user_input.strip()
        
        # Skip if empty or already processed
        if not user_input_clean:
            pass
        elif user_input_clean == st.session_state.get("last_processed", ""):
            pass  # Already processed, skip
        else:
            if user_input_clean.lower() == "exit":
                chat_bubble("Thank you for chatting. Remember, you're never alone! 💙", is_user=False)
                st.session_state.messages.append({"role": "assistant", "content": "Thank you for chatting. Remember, you're never alone! 💙"})
                time.sleep(2)
                st.stop()
            
            # Mark this input as processed
            st.session_state.last_processed = user_input_clean
            
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": user_input_clean})
            chat_bubble(user_input_clean, is_user=True)
            
            # Get and display bot response with typing animation
            with st.empty():
                st.markdown("<div class='typing-animation'>"
                            "<div class='typing-dot'></div>"
                            "<div class='typing-dot'></div>"
                            "<div class='typing-dot'></div></div>", 
                            unsafe_allow_html=True)
                response = get_best_response(user_input_clean.lower(), st.session_state.dataset)
                time.sleep(0.6)  # Simulate typing delay
            
            chat_bubble(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Text to speech (silent fail in production)
            try:
                text_to_speech(response)
            except:
                pass  # Silent fail for production
            
            # Rerun to show the new messages (form clears automatically)
            st.rerun()

if __name__ == "__main__":
    main()