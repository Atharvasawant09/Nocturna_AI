import argparse
import logging
import os
import json
from dream_analysis_system import DreamAnalysisSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dream_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the Dream Analysis System"""
    parser = argparse.ArgumentParser(description='Dream Analysis System')
    
    parser.add_argument('--data', 
                        type=str, 
                        default='dreams_dataset.csv',
                        help='Path to dreams dataset CSV file')
    
    parser.add_argument('--text_column', 
                        type=str, 
                        default='Text',
                        help='Column name containing dream text')
    
    parser.add_argument('--output', 
                        type=str, 
                        default='dream_analysis_output',
                        help='Output directory for results')
    
    parser.add_argument('--model', 
                        type=str, 
                        choices=['ml', 'lstm', 'transformer'],
                        default='lstm',
                        help='Model type to use')
    
    args = parser.parse_args()
    
    config = {
        'data_path': args.data,
        'text_column': args.text_column,
        'output_dir': args.output,
        'model_type': args.model,
        'random_seed': 42,
        'test_size': 0.2,
        'save_models': True
    }
    
    logger.info(f"Starting Dream Analysis System with config: {config}")
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    try:
        system = DreamAnalysisSystem(config)
        system.load_and_preprocess_data()
        system.prepare_features_and_split()
        system.build_and_train_models()
        results = system.evaluate_models()
        system.analyze_dream_patterns()
        
        logger.info(f"Dream analysis complete. Results available in {config['output_dir']}")
        
        if results:
            for model_name, metrics in results.items():
                print(f"\nModel: {model_name}")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"ROC AUC: {metrics['roc_auc']:.4f}")
                print("Classification Report:")
                if 'classification_report' in metrics:
                    for class_name, values in metrics['classification_report'].items():
                        if isinstance(values, dict):
                            print(f"  Class {class_name}:")
                            print(f"    Precision: {values['precision']:.4f}")
                            print(f"    Recall: {values['recall']:.4f}")
                            print(f"    F1-Score: {values['f1-score']:.4f}")
    
    except Exception as e:
        logger.error(f"Error running Dream Analysis System: {e}", exc_info=True)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()