import torch
import pandas as pd
import numpy as np
import re
import os
import pickle
import gradio as gr
from torch import nn
import json
import logging
import sys
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dream_analysis_deployment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Path configurations - Update these paths to match your file structure
MODEL_PATH = 'lstm_model.pth'  # Changed to look in current directory
VOCAB_PATH = 'vocab.pkl'       # Changed to look in current directory
OUTPUT_DIR = 'dream_analysis_output'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

class LSTMModel(nn.Module):
    """LSTM model for dream analysis"""
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, output_dim=1, n_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = hidden[-1]
        prediction = self.fc(hidden)
        return self.sigmoid(prediction)

def preprocess_text(text):
    """Clean and preprocess a single dream text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and keep only alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# A default vocabulary class in case the original is missing
class Vocabulary:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.idx = 2
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def build_vocab_from_text(self, text_list):
        for text in text_list:
            words = text.lower().split()
            for word in words:
                self.add_word(word)

def load_vocab_and_model():
    """Load the vocabulary and trained model"""
    global MODEL_PATH, VOCAB_PATH
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at: {MODEL_PATH}")
        logger.info("Searching for model file...")
        # Try to find model file in current directory or subdirectories
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.pth'):
                    MODEL_PATH = os.path.join(root, file)
                    logger.info(f"Found model file at: {MODEL_PATH}")
                    break
            if os.path.exists(MODEL_PATH):
                break
    
    if not os.path.exists(VOCAB_PATH):
        logger.warning(f"Vocabulary file not found at: {VOCAB_PATH}")
        # We'll create a dummy vocab later if needed
    
    try:
        # Load vocabulary
        vocab = None
        if os.path.exists(VOCAB_PATH):
            with open(VOCAB_PATH, 'rb') as f:
                vocab = pickle.load(f)
            logger.info(f"Loaded vocabulary with {len(vocab.word2idx)} words")
        else:
            # Create a dummy vocabulary with common words
            logger.warning("Creating dummy vocabulary")
            vocab = Vocabulary()
            common_words = ["i", "me", "my", "dream", "was", "in", "a", "the", "and", "to", "of", "that", "it", 
                           "with", "for", "on", "at", "by", "from", "up", "down", "around", "through", 
                           "over", "under", "before", "after", "during", "while", "because", "if", 
                           "then", "than", "also", "or", "but", "so", "very", "really", "just", "not"]
            for word in common_words:
                vocab.add_word(word)
            
            # Save this vocab for future use
            with open("dummy_vocab.pkl", "wb") as f:
                pickle.dump(vocab, f)
            logger.info(f"Created dummy vocabulary with {len(vocab.word2idx)} words")
        
        # Get vocab size for model initialization
        vocab_size = len(vocab.word2idx)
        logger.info(f"Using vocabulary size: {vocab_size}")
        
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at: {MODEL_PATH}")
            raise FileNotFoundError(f"Could not find model file at {MODEL_PATH}")
        
        # Initialize model with same architecture
        model = LSTMModel(vocab_size=vocab_size)
        logger.info(f"Initialized LSTM model with vocabulary size {vocab_size}")
        
        # Load trained weights
        model_state = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        # Print model state dict keys for debugging
        logger.info(f"Model state dict keys: {model_state.keys() if isinstance(model_state, dict) else 'State dict is not a dictionary'}")
        
        # Try to load the state dict
        try:
            model.load_state_dict(model_state)
        except Exception as e:
            logger.error(f"Error loading state dict: {e}")
            
            # If the state dict has a different structure, try to adapt
            if isinstance(model_state, dict) and not any(k.startswith('embedding') for k in model_state.keys()):
                logger.warning("Model state doesn't match expected structure. Attempting to adapt...")
                
                # Create a new model with parameters that might match saved model
                for hidden_dim in [64, 128, 256, 512]:
                    for embedding_dim in [50, 100, 200, 300]:
                        for n_layers in [1, 2, 3]:
                            try:
                                logger.info(f"Trying with hidden_dim={hidden_dim}, embedding_dim={embedding_dim}, n_layers={n_layers}")
                                model = LSTMModel(vocab_size=vocab_size, 
                                                  hidden_dim=hidden_dim, 
                                                  embedding_dim=embedding_dim,
                                                  n_layers=n_layers)
                                model.load_state_dict(model_state)
                                logger.info("Successfully loaded model with adjusted parameters")
                                break
                            except:
                                continue
        
        model.eval()
        logger.info("Model set to evaluation mode")
        
        return model, vocab
    
    except Exception as e:
        logger.error(f"Error loading model or vocabulary: {e}")
        logger.error(traceback.format_exc())
        
        # Try to initialize a simple model for testing
        logger.warning("Attempting to initialize a simple model for testing")
        vocab = Vocabulary()
        for i in range(1000):  # Add some dummy words
            vocab.add_word(f"word{i}")
        
        model = LSTMModel(vocab_size=len(vocab.word2idx))
        model.eval()
        
        return model, vocab

def text_to_tensor(text, vocab, max_length=100):
    """Convert preprocessed text to tensor using vocabulary"""
    processed_text = preprocess_text(text)
    words = processed_text.split()
    
    # Convert words to indices
    indices = [vocab.word2idx.get(word, vocab.word2idx['<UNK>']) for word in words]
    
    # Pad or truncate to max_length
    if len(indices) < max_length:
        indices = indices + [vocab.word2idx['<PAD>']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    
    return torch.tensor(indices).unsqueeze(0)  # Add batch dimension

def predict_anxiety_from_dream(dream_text, model, vocab, max_length=100):
    """Make prediction for anxiety indicator from dream text"""
    try:
        # Log the input for debugging
        logger.info(f"Processing dream text: {dream_text[:50]}...")
        
        # Prepare input
        tensor_input = text_to_tensor(dream_text, vocab, max_length)
        logger.info(f"Created tensor of shape {tensor_input.shape}")
        
        # Make prediction
        with torch.no_grad():
            prediction = model(tensor_input).item()
        
        logger.info(f"Raw prediction: {prediction}")
        
        # Convert to probability
        anxiety_probability = prediction * 100
        
        return anxiety_probability
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        logger.error(traceback.format_exc())
        raise

def analyze_dream(dream_text):
    """Main function to analyze dreams through Gradio interface"""
    global model, vocab
    
    try:
        # Validate input
        if not dream_text or len(dream_text.strip()) < 10:
            return {
                "Anxiety Indicator": 0.0,
                "Analysis": "Please provide a longer dream description for analysis."
            }
        
        logger.info("Analyzing dream text...")
        
        # Make sure model and vocab are loaded
        if model is None or vocab is None:
            logger.warning("Model or vocabulary not loaded. Attempting to load...")
            model, vocab = load_vocab_and_model()
        
        # Make prediction
        anxiety_score = predict_anxiety_from_dream(dream_text, model, vocab)
        logger.info(f"Anxiety score: {anxiety_score}")
        
        # Prepare analysis text based on anxiety score
        if anxiety_score > 75:
            analysis = "High anxiety indicators detected in this dream. The themes and patterns suggest significant stress or anxiety that might be affecting your well-being."
        elif anxiety_score > 50:
            analysis = "Moderate anxiety indicators found in this dream. Some elements of your dream suggest possible stress or anxiety that might be worth addressing."
        elif anxiety_score > 25:
            analysis = "Low anxiety indicators in this dream. While some elements might suggest mild stress, overall this dream doesn't show strong anxiety patterns."
        else:
            analysis = "Minimal anxiety indicators in this dream. The content and patterns of this dream don't suggest significant anxiety."
        
        return {
            "Anxiety Indicator": round(anxiety_score, 2),
            "Analysis": analysis
        }
    
    except Exception as e:
        logger.error(f"Error in dream analysis: {e}")
        logger.error(traceback.format_exc())
        return {
            "Anxiety Indicator": 0.0,
            "Analysis": f"An error occurred during analysis: {str(e)}. Please check the logs for more details."
        }

# Initialize model and vocabulary
model, vocab = None, None

# Try to load model and vocabulary at startup
try:
    logger.info("Loading model and vocabulary...")
    model, vocab = load_vocab_and_model()
    logger.info("Ready for dream analysis")
except Exception as e:
    logger.error(f"Failed to initialize: {e}")
    logger.error(traceback.format_exc())

# Create Gradio interface
examples = [
    "I was running through a maze and couldn't find the exit. Every time I thought I was close, the walls would shift and I'd be lost again.",
    "I was flying over mountains and oceans, feeling free and peaceful. The sun was shining and I could see for miles.",
    "I was back in school taking a test I hadn't studied for. Everyone else finished but I couldn't even understand the questions."
]

# Define the Gradio interface
demo = gr.Interface(
    fn=analyze_dream,
    inputs=gr.Textbox(lines=10, placeholder="Describe your dream in detail..."),
    outputs=gr.JSON(),
    title="Dream Analysis System",
    description="Enter a detailed description of your dream. Our AI will analyze it for potential anxiety indicators.",
    examples=examples,
    theme=gr.themes.Soft()
)

# Launch the app
if __name__ == "__main__":
    logger.info("Starting Gradio interface")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in current directory: {os.listdir('.')}")
    demo.launch()