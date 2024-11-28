# before running the code, make sure to install the following libraries
# pip install transformers torch pandas numpy scikit-learn tqdm

import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_preprocess_data(file_path):
    """Load and preprocess the tweet data"""
    print("Loading data...")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    df = pd.DataFrame(data)
    
    # Extract text and create basic labels
    df['text'] = df['full_text']
    
    # Create sentiment labels based on keywords
    positive_words = ['support', 'welcome', 'positive', 'good', 'great', 'excellent']
    negative_words = ['against', 'bad', 'worse', 'terrible', 'problem', 'crisis']
    
    def get_sentiment(text):
        text = str(text).lower()
        pos_count = sum(word in text for word in positive_words)
        neg_count = sum(word in text for word in negative_words)
        
        if pos_count > neg_count:
            return 2  # Positive
        elif neg_count > pos_count:
            return 0  # Negative
        else:
            return 1  # Neutral
    
    df['sentiment'] = df['text'].apply(get_sentiment)
    
    return df

def create_model():
    """Create and configure the BERT model"""
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3  # Positive, Negative, Neutral
    )
    return model

def train_model(model, train_loader, val_loader, device, num_epochs=3):
    """Train the model"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0
        train_progress = tqdm(train_loader, desc="Training")
        
        for batch in train_progress:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_progress.set_description(f"Training Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_progress = tqdm(val_loader, desc="Validation")
        
        with torch.no_grad():
            for batch in val_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                val_progress.set_description(f"Validation Loss: {loss.item():.4f}")
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses

def analyze_tweets(model, tokenizer, texts, device):
    """Analyze tweets using the trained model"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Analyzing tweets"):
            encoding = tokenizer.encode_plus(
                str(text),
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, predicted = torch.max(outputs.logits, dim=1)
            predictions.append(predicted.item())
    
    return predictions

def main():
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs('semantic_analysis_output', exist_ok=True)
    
    # Load and preprocess data
    df = load_and_preprocess_data('Election Tweets.json')
    
    # Split data
    texts = df['text'].values
    labels = df['sentiment'].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Initialize tokenizer and create datasets
    print("Initializing BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = TweetDataset(X_train, y_train, tokenizer)
    val_dataset = TweetDataset(X_val, y_val, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Create and train model
    print("Creating and training model...")
    model = create_model()
    model.to(device)
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, device)
    
    # Plot training results
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('semantic_analysis_output/training_loss.png')
    plt.close()
    
    # Analyze all tweets
    print("Analyzing all tweets...")
    predictions = analyze_tweets(model, tokenizer, df['text'].values, device)
    df['predicted_sentiment'] = predictions
    
    # Generate analysis results
    sentiment_distribution = pd.DataFrame({
        'Sentiment': ['Negative', 'Neutral', 'Positive'],
        'Count': np.bincount(predictions)
    })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=sentiment_distribution, x='Sentiment', y='Count')
    plt.title('Distribution of Predicted Sentiments')
    plt.savefig('semantic_analysis_output/sentiment_distribution.png')
    plt.close()
    
    # Save results
    analysis_results = {
        'sentiment_distribution': sentiment_distribution.to_dict(),
        'sample_predictions': df[['text', 'predicted_sentiment']].head(10).to_dict()
    }
    
    with open('semantic_analysis_output/semantic_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print("\nSemantic analysis complete! Results saved in 'semantic_analysis_output' directory:")
    print("- training_loss.png")
    print("- sentiment_distribution.png")
    print("- semantic_analysis_results.json")

if __name__ == "__main__":
    main()