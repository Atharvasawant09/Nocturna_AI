import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import torch

def evaluate_model(models, X_test, y_test, model_type='lstm', output_dir='output'):
    """Evaluate model performance on test data"""
    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_type in ['ml', 'traditional']:
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'accuracy': float(model.score(X_test, y_test)),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
            }
                
    else:  # LSTM or transformer
        model_name = list(models.keys())[0]
        model = models[model_name]
        
        model.eval()
        model.to(device)
        
        with torch.no_grad():
            if model_type == 'lstm':
                X_test = X_test.to(device)
                y_pred_proba = model(X_test).squeeze().detach().cpu().numpy()
            elif model_type == 'transformer':
                X_test = {k: v.to(device) for k, v in X_test.items()}
                outputs = model(**X_test)
                y_pred_proba = torch.sigmoid(outputs.logits).squeeze().detach().cpu().numpy()
            
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            results[model_name] = {
                'accuracy': float((y_pred == y_test).mean()),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
            }
    
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    
    def convert_to_serializable(obj):
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
    
    serializable_results = convert_to_serializable(results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    return results

def create_evaluation_plots(models, X_test, y_test, results, model_type='lstm', output_dir='output'):
    """Create evaluation plots for model performance"""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for model_name, model in models.items():
        model_results = results[model_name]
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = np.array(model_results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Condition', 'Has Condition'],
                   yticklabels=['No Condition', 'Has Condition'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{model_name}_confusion_matrix.png'))
        plt.close()
        
        # ROC Curve
        plt.figure(figsize=(8, 6))
        
        if model_type in ['ml', 'traditional']:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            model.eval()
            model.to(device)
            with torch.no_grad():
                if model_type == 'lstm':
                    X_test = X_test.to(device)
                    y_pred_proba = model(X_test).squeeze().detach().cpu().numpy()
                elif model_type == 'transformer':
                    X_test = {k: v.to(device) for k, v in X_test.items()}
                    outputs = model(**X_test)
                    y_pred_proba = torch.sigmoid(outputs.logits).squeeze().detach().cpu().numpy()
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        plt.plot(fpr, tpr, label=f'AUC = {model_results["roc_auc"]:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(plots_dir, f'{model_name}_roc_curve.png'))
        plt.close()