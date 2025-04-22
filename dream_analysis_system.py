import pandas as pd
import numpy as np
import re
import os
import json
import logging
import torch

class DreamAnalysisSystem:
    """Complete system for analyzing dreams"""
    
    def __init__(self, config=None):
        """Initialize with configuration parameters"""
        self.config = config or {
            'data_path': 'dreams_dataset.csv',
            'text_column': 'Text',
            'id_column': 'ID',
            'output_dir': 'dream_analysis_output',
            'random_seed': 42,
            'test_size': 0.2,
            'model_type': 'lstm',
            'save_models': True
        }
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        self.raw_data = None
        self.processed_data = None
        self.feature_extractor = None
        self.models = {}
        self.evaluation_results = {}
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("dream_analysis.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dream dataset"""
        from preprocess import load_data, clean_and_process_data
        
        self.logger.info(f"Loading data from {self.config['data_path']}")
        
        try:
            self.raw_data = load_data(self.config['data_path'])
            self.logger.info(f"Loaded {len(self.raw_data)} dream records")
            
            self.processed_data = clean_and_process_data(
                self.raw_data, 
                text_column=self.config['text_column'],
                output_dir=self.config['output_dir']
            )
            
            self.condition_to_predict = 'anxiety_indicator'
            
            self.logger.info("Preprocessing complete")
            return self.processed_data
            
        except Exception as e:
            self.logger.error(f"Error during data loading and preprocessing: {e}")
            raise
    
    def prepare_features_and_split(self):
        """Prepare features and split data into training and testing sets"""
        from preprocess import prepare_features
        
        if self.processed_data is None:
            self.logger.error("No processed data available. Call load_and_preprocess_data first.")
            return
        
        try:
            self.logger.info("Preparing features and splitting data")
            
            result = prepare_features(
                self.processed_data,
                self.condition_to_predict,
                model_type=self.config['model_type'],
                test_size=self.config['test_size'],
                random_seed=self.config['random_seed']
            )
            
            self.X_train = result['X_train']
            self.X_test = result['X_test']
            self.y_train = result['y_train']
            self.y_test = result['y_test']
            self.feature_extractor = result['feature_extractor']
            
            if self.config['model_type'] == 'lstm':
                self.max_length = result.get('max_length')
                self.vocab_size = result.get('vocab_size')
            
            self.logger.info("Feature preparation complete")
            return self.X_train, self.y_train, self.X_test, self.y_test
            
        except Exception as e:
            self.logger.error(f"Error during feature preparation: {e}")
            raise
    
    def build_and_train_models(self):
        """Build and train the selected model type"""
        from model import build_and_train_model
        
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            self.logger.error("Training data not prepared. Call prepare_features_and_split first.")
            return
        
        try:
            self.logger.info(f"Building and training {self.config['model_type']} model")
            
            result = build_and_train_model(
                self.X_train, 
                self.y_train,
                model_type=self.config['model_type'],
                vocab_size=getattr(self, 'vocab_size', None),
                max_length=getattr(self, 'max_length', None)
            )
            
            self.models = result['models']
            
            if 'training_history' in result:
                self.training_history = result['training_history']
            
            if self.config['save_models']:
                self._save_models()
            
            self.logger.info("Model training complete")
            return self.models
            
        except Exception as e:
            self.logger.error(f"Error during model building and training: {e}")
            raise
    
    def evaluate_models(self):
        """Evaluate trained models on test data"""
        from evaluate import evaluate_model, create_evaluation_plots
        
        if not self.models:
            self.logger.error("No trained models available. Call build_and_train_models first.")
            return
        
        try:
            self.logger.info("Evaluating model performance")
            
            results = evaluate_model(
                self.models,
                self.X_test,
                self.y_test,
                model_type=self.config['model_type'],
                output_dir=self.config['output_dir']
            )
            
            self.evaluation_results = results
            
            create_evaluation_plots(
                self.models,
                self.X_test,
                self.y_test,
                results,
                model_type=self.config['model_type'],
                output_dir=self.config['output_dir']
            )
            
            self.logger.info("Evaluation complete")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {e}")
            raise
    
    def analyze_dream_patterns(self):
        """Analyze patterns in dreams related to mental health indicators"""
        from utils import analyze_patterns
        
        if self.processed_data is None:
            self.logger.error("No processed data available. Call load_and_preprocess_data first.")
            return
        
        try:
            self.logger.info("Analyzing dream patterns")
            
            analysis_results = analyze_patterns(
                self.processed_data,
                self.condition_to_predict,
                output_dir=self.config['output_dir']
            )
            
            self.logger.info("Pattern analysis complete")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error during pattern analysis: {e}")
            raise
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            import joblib
            import torch
            
            model_dir = os.path.join(self.config['output_dir'], 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            if self.config['model_type'] in ['ml', 'traditional']:
                for name, model in self.models.items():
                    model_path = os.path.join(model_dir, f"{name}_model.joblib")
                    joblib.dump(model, model_path)
                    self.logger.info(f"Saved {name} model to {model_path}")
                
                extractor_path = os.path.join(model_dir, "feature_extractor.joblib")
                joblib.dump(self.feature_extractor, extractor_path)
                self.logger.info(f"Saved feature extractor to {extractor_path}")
                
            elif self.config['model_type'] in ['lstm', 'transformer']:
                for name, model in self.models.items():
                    model_path = os.path.join(model_dir, f"{name}_model.pth")
                    torch.save(model.state_dict(), model_path)
                    self.logger.info(f"Saved {name} model to {model_path}")
                
                if hasattr(self, 'feature_extractor'):
                    if self.config['model_type'] == 'lstm':
                        import pickle
                        vocab_path = os.path.join(model_dir, "vocab.pkl")
                        with open(vocab_path, 'wb') as f:
                            pickle.dump(self.feature_extractor, f)
                        self.logger.info(f"Saved vocab to {vocab_path}")
                    elif self.config['model_type'] == 'transformer':
                        self.feature_extractor.save_pretrained(model_dir)
                        self.logger.info(f"Saved tokenizer to {model_dir}")
            
            self.logger.info("All models saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            raise
    
    def predict_dream(self, dream_text):
        """Make predictions for a new dream text"""
        from utils import preprocess_single_dream, make_prediction
        
        if not self.models:
            self.logger.error("No trained models available. Call build_and_train_models first.")
            return
        
        try:
            processed_text = preprocess_single_dream(dream_text)
            prediction = make_prediction(
                processed_text,
                self.models,
                self.feature_extractor,
                model_type=self.config['model_type'],
                max_length=getattr(self, 'max_length', None)
            )
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting dream: {e}")
            raise