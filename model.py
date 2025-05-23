import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from transformers import BertForSequenceClassification

def build_traditional_ml_models():
    """Build traditional machine learning models"""
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    return {'random_forest': rf_model, 'neural_network': nn_model}

class LSTMModel(nn.Module):
    """LSTM model for text classification"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, output_dim=1, n_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        packed_output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate the final hidden states from both directions
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Output shape will be [batch_size, output_dim]
        output = self.fc(hidden)
        
        return output

def build_lstm_model(vocab_size, max_length):
    """Build and return LSTM model configuration"""
    return LSTMModel(vocab_size=vocab_size, embedding_dim=128, hidden_dim=64, output_dim=1)

def build_transformer_model():
    """Build a transformer-based model using BERT"""
    return BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

def train_pytorch_model(model, X_train, y_train, epochs=10, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train a PyTorch model"""
    criterion = nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss to handle logits directly
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Move data to device
    if isinstance(X_train, dict):
        X_train = {k: v.to(device) for k, v in X_train.items()}
    else:
        X_train = X_train.to(device)
    
    # Convert y_train to 2D tensor with shape [batch_size, 1]
    y_train = torch.tensor(y_train, dtype=torch.float).view(-1, 1).to(device)

    # Create dataset and dataloader
    if isinstance(X_train, dict):
        dataset = torch.utils.data.TensorDataset(X_train['input_ids'], X_train['attention_mask'], y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {'loss': [], 'accuracy': []}

    model.to(device)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_data in dataloader:
            optimizer.zero_grad()
            
            if isinstance(X_train, dict):
                # For transformer models with attention mask
                batch_X, batch_attention, batch_y = batch_data
                outputs = model(input_ids=batch_X, attention_mask=batch_attention)
                # Get logits from transformer output
                pred_logits = outputs.logits.view(-1, 1)
                loss = criterion(pred_logits, batch_y)
                # For accuracy calculation (apply sigmoid since we're using BCEWithLogitsLoss)
                predicted = (torch.sigmoid(pred_logits) > 0.5).float()
            else:
                # For LSTM model
                batch_X, batch_y = batch_data
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                # For accuracy calculation
                predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_loss = epoch_loss / len(dataloader)
        epoch_acc = correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    return model, history

def build_and_train_model(X_train, y_train, model_type='lstm', vocab_size=None, max_length=None):
    """Build and train models based on specified type"""
    result = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_type in ['ml', 'traditional']:
        models = build_traditional_ml_models()
        for name, model in models.items():
            model.fit(X_train, y_train)
        result['models'] = models
        
    elif model_type == 'lstm':
        model = build_lstm_model(vocab_size, max_length)
        model, history = train_pytorch_model(model, X_train, y_train, epochs=10, batch_size=32, device=device)
        result['models'] = {'lstm': model}
        result['training_history'] = history
        
    elif model_type == 'transformer':
        model = build_transformer_model()
        model, history = train_pytorch_model(model, X_train, y_train, epochs=3, batch_size=16, device=device)
        result['models'] = {'transformer': model}
        result['training_history'] = history

    return result