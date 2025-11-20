import json
import os
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def train_svm_model():
    print("Loading dataset...")
    # Load intents
    try:
        with open('../datasets/intents.json', 'r', encoding='utf-8') as f:
            intents = json.load(f)
    except FileNotFoundError:
        # Try local path if running from models dir
        with open('datasets/intents.json', 'r', encoding='utf-8') as f:
            intents = json.load(f)

    # Prepare training data
    patterns = []
    tags = []
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if pattern.strip():
                patterns.append(pattern)
                tags.append(intent['tag'])
    
    print(f"Training on {len(patterns)} patterns across {len(set(tags))} tags.")

    # Create pipeline: TF-IDF -> SVM
    # probability=True is needed for confidence scores
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', SVC(kernel='linear', probability=True))
    ])

    # Train model
    print("Training model...")
    pipeline.fit(patterns, tags)
    
    # Save model
    model_path = 'svm_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"Model saved to {model_path}")
    
    # Test a few examples
    test_phrases = ["Hello there", "I feel sad", "Goodbye", "Tell me a joke"]
    print("\nTest Predictions:")
    for phrase in test_phrases:
        prediction = pipeline.predict([phrase])[0]
        proba = pipeline.predict_proba([phrase]).max()
        print(f"'{phrase}' -> {prediction} ({proba:.2f})")

if __name__ == "__main__":
    train_svm_model()
