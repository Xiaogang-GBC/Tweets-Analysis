# this code runs about 60 minutes on a M1 Macbook Air
import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# use class to encapsulate all the sentiment analysis methods
class TweetDataset(Dataset):
    # initialize the dataset with the texts, labels, tokenizer and max_length
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    # return the length of the dataset
    def __len__(self):
        return len(self.texts)

    # return the item at the given index
    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            # return the input_ids, attention_mask and labels
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# use class to encapsulate all the sentiment analysis methods
class SentimentAnalyzer:
    # initialize the analyzer with the file path
    def __init__(self, file_path):
        """Initialize analyzer with data and create output directory"""
        self.output_dir = 'export/sentiment'
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.load_data(file_path)

    def load_data(self, file_path):
        """Load and preprocess tweet data"""
        print("Loading tweet data...")
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.df = pd.DataFrame(data)
        print(f"Loaded {len(self.df)} tweets")

    def create_labels(self):
        """Create initial sentiment labels using keyword matching"""
        print("Creating initial sentiment labels...")
        
        # Define positive and negative words
        positive_words = {'support', 'welcome', 'positive', 'good', 'great', 'excellent'}
        negative_words = {'against', 'bad', 'worse', 'terrible', 'problem', 'crisis'}
        
        # Function to get sentiment based on keyword matching
        def get_sentiment(text):
            text = str(text).lower()
            pos_count = sum(word in text for word in positive_words)
            neg_count = sum(word in text for word in negative_words)
            
            if pos_count > neg_count:
                return 2  # Positive
            elif neg_count > pos_count:
                return 0  # Negative
            return 1  # Neutral
        
        self.df['sentiment'] = self.df['full_text'].apply(get_sentiment)

    def train_model(self, num_epochs=3):
        """Train the sentiment analysis model"""
        print("Preparing for training...")
        
        # Prepare data
        texts = self.df['full_text'].values
        labels = self.df['sentiment'].values
        
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # use the AutoModelForSequenceClassification model
        model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=3
        ).to(self.device)
        
        # Create datasets and dataloaders
        train_dataset = TweetDataset(X_train, y_train, tokenizer)
        val_dataset = TweetDataset(X_val, y_val, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        train_losses = []
        val_losses = []
        
        # Training loop
        print("Starting training...")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            model.train()
            train_loss = 0
            train_progress = tqdm(train_loader, desc="Training")
            
            # Loop through each batch
            for batch in train_progress:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    labels=batch['labels'].to(self.device)
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_progress.set_description(f"Training Loss: {loss.item():.4f}")
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0
            predictions = []
            
            # Loop through each batch
            with torch.no_grad():
                # Loop through each batch of validation
                for batch in tqdm(val_loader, desc="Validation"):
                    # Forward pass
                    outputs = model(
                        input_ids=batch['input_ids'].to(self.device),
                        attention_mask=batch['attention_mask'].to(self.device),
                        labels=batch['labels'].to(self.device)
                    )
                    
                    val_loss += outputs.loss.item()
                    _, preds = torch.max(outputs.logits, dim=1)
                    predictions.extend(preds.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Average Training Loss: {avg_train_loss:.4f}")
            print(f"Average Validation Loss: {avg_val_loss:.4f}")
        
        # Plot training progress
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{self.output_dir}/training_loss.png')
        plt.close()
        
        # Plot sentiment distribution
        plt.figure(figsize=(10, 6))
        sentiment_dist = pd.Series(predictions).value_counts().sort_index()
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        plt.bar(sentiment_labels, sentiment_dist)
        plt.title('Distribution of Predicted Sentiments')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_distribution.png')
        plt.close()
        
        # Save predictions and model statistics
        results = {
            'sentiment_distribution': dict(zip(sentiment_labels, sentiment_dist.tolist())),
            'final_training_loss': train_losses[-1],
            'final_validation_loss': val_losses[-1],
        }
        
        with open(f'{self.output_dir}/sentiment_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Sentiment analysis complete. Check the 'export/sentiment' directory for results.")
        return model, tokenizer

# run the sentiment analysis
if __name__ == "__main__":
    analyzer = SentimentAnalyzer('Election Tweets.json')
    analyzer.create_labels()
    model, tokenizer = analyzer.train_model()
