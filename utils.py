import re
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
import logging
import torch

logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean text by removing URLs, special characters, and converting to lowercase"""
    if isinstance(text, str):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s.,!?]', '', text)
        return text.lower().strip()
    return ''

def detect_anxiety(text):
    """Detect anxiety-related keywords in text"""
    anxiety_keywords = ['anxious', 'worry', 'afraid', 'fear', 'panic', 'stress', 
                        'scared', 'nervous', 'terror', 'nightmare', 'chase']
    return 1 if isinstance(text, str) and any(keyword in text.lower() for keyword in anxiety_keywords) else 0

def detect_depression(text):
    """Detect depression-related keywords in text"""
    depression_keywords = ['sad', 'depressed', 'empty', 'hopeless', 'worthless', 
                           'lonely', 'miserable', 'unhappy', 'crying', 'tears']
    return 1 if isinstance(text, str) and any(keyword in text.lower() for keyword in depression_keywords) else 0

def generate_wordcloud(text, output_path, title=None):
    """Generate and save wordcloud from text"""
    stop_words = set(stopwords.words('english'))
    
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        stopwords=stop_words,
        max_words=100
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Wordcloud saved to {output_path}")

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

def preprocess_single_dream(dream_text):
    """Preprocess a single dream text for prediction"""
    return clean_text(dream_text)

def make_prediction(processed_text, models, feature_extractor, model_type='lstm', max_length=None):
    """Make prediction for a single dream text"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = list(models.keys())[0]
    model = models[model_name]
    model.eval()
    
    with torch.no_grad():
        if model_type in ['ml', 'traditional']:
            X = feature_extractor.transform([processed_text])
            return model.predict_proba(X)[:, 1][0]
        elif model_type == 'lstm':
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            X = pad_sequences(feature_extractor.texts_to_sequences([processed_text]), maxlen=max_length)
            X = torch.tensor(X, dtype=torch.long).to(device)
            return model(X).squeeze().detach().cpu().numpy()
        elif model_type == 'transformer':
            X = feature_extractor(processed_text, truncation=True, padding=True, max_length=128, return_tensors='pt')
            X = {k: v.to(device) for k, v in X.items()}
            outputs = model(**X)
            return torch.sigmoid(outputs.logits).squeeze().detach().cpu().numpy()

def analyze_patterns(df, condition, output_dir='output'):
    """Analyze patterns in dreams related to mental health indicators"""
    analysis_results = {}
    
    positive_df = df[df[condition] == 1]
    negative_df = df[df[condition] == 0]
    
    positive_text = ' '.join(positive_df['clean_text'].tolist())
    negative_text = ' '.join(negative_df['clean_text'].tolist())
    
    generate_wordcloud(positive_text, os.path.join(output_dir, f'{condition}_positive_wordcloud.png'), 
                       f"Word Cloud for Dreams with {condition.capitalize()}")
    generate_wordcloud(negative_text, os.path.join(output_dir, f'{condition}_negative_wordcloud.png'), 
                       f"Word Cloud for Dreams without {condition.capitalize()}")
    
    theme_columns = [col for col in df.columns if col.startswith('theme_')]
    positive_themes = positive_df[theme_columns].mean()
    negative_themes = negative_df[theme_columns].mean()
    
    analysis_results['positive_themes'] = positive_themes.to_dict()
    analysis_results['negative_themes'] = negative_themes.to_dict()
    
    with open(os.path.join(output_dir, 'pattern_analysis.json'), 'w') as f:
        json.dump(analysis_results, f, indent=4)
    
    return analysis_results