import pandas as pd
import numpy as np
import re
import os
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from nltk.sentiment import SentimentIntensityAnalyzer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
# Ensure punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Download other necessary NLTK data
nltk.download('punkt_tab', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)

def load_data(data_path):
    """Load dream dataset from CSV file"""
    return pd.read_csv(data_path)

def clean_text(text):
    """Clean and normalize text data"""
    if isinstance(text, str):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s.,!?]', '', text)
        return text.lower().strip()
    return ''

def extract_dream_features_nltk(text):
    """Extract key elements from dream text using NLTK"""
    if not isinstance(text, str) or not text.strip():
        return {'entities': [], 'emotions': [], 'themes': {}}
    
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    
    emotion_words = [word for word, pos in tagged if pos.startswith('JJ')]
    entities = [word for word, pos in tagged if pos in ['NNP', 'NN']]
    
    themes = {
        'falling': 1 if re.search(r'\b(fall(ing)?|fell)\b', text.lower()) else 0,
        'flying': 1 if re.search(r'\b(fly(ing)?|flew)\b', text.lower()) else 0,
        'chase': 1 if re.search(r'\b(chas(e|ing|ed))\b', text.lower()) else 0,
        'death': 1 if re.search(r'\b(death|dying|dead|die)\b', text.lower()) else 0,
        'water': 1 if re.search(r'\b(water|ocean|sea|swim)\b', text.lower()) else 0,
        'family': 1 if re.search(r'\b(family|mother|father|sister|brother|parent)\b', text.lower()) else 0,
        'school': 1 if re.search(r'\b(school|class|teacher|student)\b', text.lower()) else 0,
        'teeth': 1 if re.search(r'\b(teeth|tooth)\b', text.lower()) else 0,
        'naked': 1 if re.search(r'\b(naked|nude|cloth(es)?)\b', text.lower()) else 0,
        'test': 1 if re.search(r'\b(test|exam)\b', text.lower()) else 0,
    }
    
    return {'entities': entities, 'emotions': emotion_words, 'themes': themes}

def detect_anxiety(text):
    """Detect anxiety/stress keywords in text"""
    anxiety_keywords = ['anxious', 'worry', 'afraid', 'fear', 'panic', 'stress', 
                        'scared', 'nervous', 'terror', 'nightmare', 'chase']
    return 1 if isinstance(text, str) and any(keyword in text.lower() for keyword in anxiety_keywords) else 0

def detect_depression(text):
    """Detect depression keywords in text"""
    depression_keywords = ['sad', 'depressed', 'empty', 'hopeless', 'worthless', 
                           'lonely', 'miserable', 'unhappy', 'crying', 'tears']
    return 1 if isinstance(text, str) and any(keyword in text.lower() for keyword in depression_keywords) else 0

def clean_and_process_data(df, text_column='Text', output_dir='output'):
    """Clean and process the dream dataset"""
    processed_df = df.copy()
    
    processed_df['clean_text'] = processed_df[text_column].apply(clean_text)
    processed_df = processed_df[processed_df['clean_text'].str.len() > 0]
    
    processed_df['features'] = processed_df['clean_text'].apply(extract_dream_features_nltk)
    
    for theme in ['falling', 'flying', 'chase', 'death', 'water', 'family', 'school', 'teeth', 'naked', 'test']:
        processed_df[f'theme_{theme}'] = processed_df['features'].apply(lambda x: x['themes'].get(theme, 0))
    
    sia = SentimentIntensityAnalyzer()
    processed_df['sentiment'] = processed_df['clean_text'].apply(lambda x: sia.polarity_scores(x))
    processed_df['negative_score'] = processed_df['sentiment'].apply(lambda x: x['neg'])
    processed_df['positive_score'] = processed_df['sentiment'].apply(lambda x: x['pos'])
    
    processed_df['anxiety_indicator'] = processed_df['clean_text'].apply(detect_anxiety)
    processed_df['depression_indicator'] = processed_df['clean_text'].apply(detect_depression)
    processed_df['high_negative'] = (processed_df['negative_score'] > 0.3).astype(int)
    
    processed_path = os.path.join(output_dir, 'processed_dreams.csv')
    processed_df.to_csv(processed_path, index=False)
    
    return processed_df

def prepare_features(df, target_column, model_type='lstm', test_size=0.2, random_seed=42):
    """Prepare features and split data into training and testing sets"""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed)
    
    y_train = train_df[target_column].values
    y_test = test_df[target_column].values
    
    result = {'y_train': y_train, 'y_test': y_test, 'train_df': train_df, 'test_df': test_df}
    
    if model_type in ['ml', 'traditional']:
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train = tfidf.fit_transform(train_df['clean_text'])
        X_test = tfidf.transform(test_df['clean_text'])
        
        result['feature_extractor'] = tfidf
        result['X_train'] = X_train
        result['X_test'] = X_test
        
    elif model_type == 'lstm':
        max_words = 10000
        max_length = 200
        
        # Use torchtext for tokenization
        tokenizer = get_tokenizer('basic_english')
        
        def yield_tokens(data_iter):
            for text in data_iter:
                yield tokenizer(text)
        
        # Build vocabulary
        vocab = build_vocab_from_iterator(yield_tokens(train_df['clean_text']), 
                                        max_tokens=max_words, 
                                        specials=['<pad>', '<unk>'])
        vocab.set_default_index(vocab['<unk>'])
        
        # Convert texts to sequences
        X_train = [torch.tensor([vocab[token] for token in tokenizer(text)] + [vocab['<pad>']] * max(0, max_length - len(tokenizer(text))), 
                                dtype=torch.long)[:max_length] for text in train_df['clean_text']]
        X_test = [torch.tensor([vocab[token] for token in tokenizer(text)] + [vocab['<pad>']] * max(0, max_length - len(tokenizer(text))), 
                               dtype=torch.long)[:max_length] for text in test_df['clean_text']]
        
        # Pad sequences to ensure uniform length
        X_train = torch.stack([x[:max_length] if len(x) >= max_length else torch.cat([x, torch.full((max_length - len(x),), vocab['<pad>'], dtype=torch.long)]) for x in X_train])
        X_test = torch.stack([x[:max_length] if len(x) >= max_length else torch.cat([x, torch.full((max_length - len(x),), vocab['<pad>'], dtype=torch.long)]) for x in X_test])
        
        result['feature_extractor'] = vocab  # Store vocab instead of tokenizer
        result['max_length'] = max_length
        result['vocab_size'] = len(vocab)
        result['X_train'] = X_train
        result['X_test'] = X_test
        
    elif model_type == 'transformer':
        from transformers import BertTokenizer
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        def encode_texts(texts, max_length=128):
            return tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        
        X_train = encode_texts(train_df['clean_text'].tolist())
        X_test = encode_texts(test_df['clean_text'].tolist())
        
        result['feature_extractor'] = tokenizer
        result['X_train'] = X_train
        result['X_test'] = X_test
    
    return result