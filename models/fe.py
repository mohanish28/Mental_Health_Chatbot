# app.py
import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import process
from speech_recognition import Recognizer, Microphone, UnknownValueError, RequestError
import pyttsx3
import time
import json
from datasets import load_dataset
import random

# Custom CSS for animations and styling
st.markdown("""
<style>
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.chat-message {
    padding: 1.5rem;
    border-radius: 1rem;
    margin: 1rem 0;
    animation: fadeIn 0.5s ease-in;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}

.user-message {
    background: linear-gradient(145deg, #e3f2fd, #bbdefb);
    margin-left: 20%;
}

.bot-message {
    background: linear-gradient(145deg, #f0f4c3, #dce775);
    margin-right: 20%;
}

.status-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 8px;
}

.typing-animation {
    display: inline-block;
    position: relative;
}

.typing-dot {
    width: 6px;
    height: 6px;
    background: #666;
    border-radius: 50%;
    display: inline-block;
    margin: 0 2px;
    animation: typing 1.4s infinite;
}

@keyframes typing {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}
</style>
""", unsafe_allow_html=True)

def load_datasets():
    """Load and combine datasets"""
    try:
        # Load Hugging Face dataset
        ds = load_dataset("Amod/mental_health_counseling_conversations")
        df = pd.DataFrame(ds["train"])
        hf_dataset = {row["Context"].lower(): row["Response"] for _, row in df.iterrows()}
        
        # Load local JSON dataset
        with open("C:\\Users\\mohan\\OneDrive\\Desktop\\cu hakathon\\intents.json", "r") as f:
            json_data = json.load(f)
            json_dataset = {}
            for intent in json_data["intents"]:
                for pattern in intent["patterns"]:
                    json_dataset[pattern.lower()] = random.choice(intent["responses"])
        
        return {**hf_dataset, **json_dataset}
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return {}

def get_best_response(user_input, dataset):
    """Get best match using fuzzywuzzy"""
    if not user_input:
        return "I'm here to listen. Please tell me how you're feeling."
    
    best_match, score = process.extractOne(user_input, dataset.keys())
    return dataset[best_match] if score > 70 else random.choice([
        "Could you elaborate on that?",
        "I'm here to support you. Tell me more.",
        "How does that make you feel?",
        "Let's explore that together."
    ])

def recognize_speech():
    """Speech to text conversion"""
    recognizer = Recognizer()
    with Microphone() as source:
        with st.status("Listening...", state="running"):
            try:
                audio = recognizer.listen(source, timeout=5)
                return recognizer.recognize_google(audio).lower()
            except UnknownValueError:
                st.warning("Could not understand audio")
            except RequestError:
                st.error("Speech service unavailable")
            except Exception as e:
                st.error(f"Error: {e}")
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
    st.title("🤖 Mindful Companion")
    st.subheader("Your 24/7 Mental Health Support")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.dataset = load_datasets()
    
    # Sidebar with animations
    with st.sidebar:
        st.markdown("###   Status")
        st.markdown("<div class='status-indicator' style='background: #4CAF50;'></div> System Active", unsafe_allow_html=True)
        st.markdown("### 📌 Tips")
        st.info("""
        - Speak naturally like you're talking to a friend
        - Try both text and voice input
        - Type 'exit' to end conversation
        """)
    
    # Chat interface
    for message in st.session_state.messages:
        chat_bubble(message["content"], is_user=message["role"] == "user")
    
    # Input options
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input("Type your message...", key="input")
    with col2:
        if st.button("🎤 Voice", use_container_width=True):
            user_input = recognize_speech()
    
    if user_input:
        if user_input.lower() == "exit":
            with st.spinner("Wrapping up..."):
                time.sleep(1)
                st.success("Thank you for chatting. Remember, you're never alone! 💙")
                st.stop()
        
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        chat_bubble(user_input, is_user=True)
        
        # Get and display bot response
        with st.spinner(""):
            with st.empty():
                st.markdown("<div class='typing-animation'>"
                            "<div class='typing-dot'></div>"
                            "<div class='typing-dot'></div>"
                            "<div class='typing-dot'></div></div>", 
                            unsafe_allow_html=True)
                response = get_best_response(user_input.lower(), st.session_state.dataset)
                time.sleep(0.5)  # Simulate typing delay
            
            chat_bubble(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            text_to_speech(response)

if __name__ == "__main__":
    main()