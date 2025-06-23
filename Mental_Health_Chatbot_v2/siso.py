# app.py
import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import process
import speech_recognition as sr
import time
import json
from datasets import load_dataset
import random
import requests
from pydub import AudioSegment
from pydub.playback import play
import os
from transformers import pipeline
import torch

# Load summarizer
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    revision="a4f8f3e",
    device=0 if torch.cuda.is_available() else -1
)

# Custom CSS for styling
st.markdown("""
<style>
body {
    background-color: #eef3f8;
    font-family: 'Segoe UI', sans-serif;
}
.chat-message {
    padding: 1.25rem;
    border-radius: 1rem;
    margin: 1rem auto;
    max-width: 75%;
    animation: fadeIn 0.5s ease-in;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    color: #333;
}
.user-message {
    background: #e3f2fd;
    text-align: right;
    border: 2px solid #90caf9;
    color: #1565c0;
}
.bot-message {
    background: #fff3e0;
    text-align: left;
    border: 2px solid #ffb74d;
    color: #5d4037;
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
.chat-container {
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    min-height: 70vh;
}
.fixed-input {
    position: fixed;
    bottom: 1rem;
    left: 50%;
    transform: translateX(-50%);
    width: 60%;
    z-index: 10;
}
</style>
""", unsafe_allow_html=True)

# Load datasets
def load_datasets():
    try:
        ds = load_dataset("Amod/mental_health_counseling_conversations")
        df = pd.DataFrame(ds["train"])
        hf_dataset = {row["Context"].lower(): row["Response"] for _, row in df.iterrows()}

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

# Fuzzy matching for responses
def get_best_response(user_input, dataset):
    if not user_input:
        return "I'm here to listen. Please tell me how you're feeling."

    best_match, score = process.extractOne(user_input, dataset.keys())
    if score > 70:
        full_response = dataset[best_match]
        if len(full_response.split()) <= 30:
            return full_response
        else:
            summary = summarizer(full_response, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
            return summary
    else:
        return random.choice([
            "Could you elaborate on that?",
            "I'm here to support you. Tell me more.",
            "How does that make you feel?",
            "Let's explore that together."
        ])

# Voice input
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            st.info("🎙 Listening...")
            audio = recognizer.listen(source)
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return ""

# ElevenLabs TTS with dynamic pitch & tone

def text_to_speech(text, gender="female"):
    ELEVEN_API_KEY = "sk_3802b542233cf79dae984a318fafc50c1acdf1c20a962a31"
    voices = {
        "female": "cNYrMw9glwJZXR8RwbuR",
        "male": "t0jbNlBVZ17f02VDIeMI"
    }
    VOICE_ID = voices.get(gender, voices["female"])

    tone_settings = {
        "stability": 0.3 if "?" in text or "!" in text else 0.6,
        "similarity_boost": 0.75
    }

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_API_KEY
    }

    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": tone_settings
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            with open("response.mp3", "wb") as f:
                f.write(response.content)
            audio = AudioSegment.from_mp3("response.mp3")
            play(audio)
            os.remove("response.mp3")
        else:
            st.warning("TTS API error: " + response.text)
    except Exception as e:
        st.warning(f"Error using ElevenLabs: {e}")

# Chat bubble renderer
def chat_bubble(message, is_user=False):
    css_class = "user-message" if is_user else "bot-message"
    st.markdown(f"<div class='chat-message {css_class}'>{message}</div>", unsafe_allow_html=True)

# Main function
def main():
    st.title("🧘 Mindful Companion")
    st.subheader("Your 24/7 Mental Wellness Partner")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.dataset = load_datasets()
        st.session_state.gender = "female"

    with st.sidebar:
        st.markdown("### System Status")
        st.markdown("<div class='status-indicator' style='background: #4CAF50;'></div> Online", unsafe_allow_html=True)
        selected_gender = st.radio("Choose Voice:", ["female", "male"])
        st.session_state.gender = selected_gender
        st.info("""
        💡 Tips:
        - Chat like you're talking to a friend
        - Try using voice
        - Type 'exit' to finish
        """)

    for message in st.session_state.messages:
        chat_bubble(message["content"], is_user=message["role"] == "user")

    with st.container():
        user_input = st.text_input("Type your message...", key="input", label_visibility="collapsed")
        voice_button = st.button("🎤 Voice")

    if voice_button:
        user_input = recognize_speech()

    if user_input:
        if user_input.lower() == "exit":
            st.success("Session ended. Take care of yourself! 💙")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": user_input})
        chat_bubble(user_input, is_user=True)

        with st.spinner("Thinking..."):
            response = get_best_response(user_input.lower(), st.session_state.dataset)
            chat_bubble(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            text_to_speech(response, gender=st.session_state.gender)

if __name__ == "__main__":
    main()
