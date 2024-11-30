
import json
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os
from time import time

class EfficientSentimentAnalyzer:
    def __init__(self, file_path):
        self.output_dir = 'export/sentiment'
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_data(file_path)

    def load_data(self, file_path):
        print("Loading tweet data...")
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.df = pd.DataFrame(data)
        print(f"Loaded {len(self.df)} tweets")

    def detect_sarcasm(self, text):
        """Detect potential sarcasm using pattern matching"""
        sarcasm_patterns = [
            r'yeah right',
            r'sure+ly',
            r'obviously',
            r'clearly',
            r'of course',
            r'(!+\?|\?+!)',  # Mixed punctuation
            r'\.{3,}',       # Multiple periods
            r'"[^"]*"',      # Quoted text
            r'(?i)great job',
            r'(?i)well done',
            r'(?i)how nice',
            r'(?i)just perfect',
            r'(?i)exactly what we need'
        ]
        text = str(text).lower()
        return any(re.search(pattern, text) for pattern in sarcasm_patterns)

    def analyze_sentiment(self):
        """Analyze sentiment with sarcasm detection and political context"""
        print("Analyzing sentiment...")
        start_time = time()

        def get_refined_sentiment(text):
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            is_sarcastic = self.detect_sarcasm(text)

            # Adjust polarity based on sarcasm
            if is_sarcastic:
                polarity = -polarity

            # Custom thresholds for political content
            if subjectivity > 0.5:
                if polarity > 0.2:
                    return 'positive'
                elif polarity < -0.2:
                    return 'negative'
                return 'neutral'
            else:
                if polarity > 0.1:
                    return 'positive'
                elif polarity < -0.1:
                    return 'negative'
                return 'neutral'

        # Apply sentiment analysis
        self.df['sentiment'] = self.df['full_text'].apply(get_refined_sentiment)
        self.df['is_sarcastic'] = self.df['full_text'].apply(self.detect_sarcasm)
        
        # Calculate statistics
        sentiment_counts = self.df['sentiment'].value_counts()
        sentiment_percentages = (sentiment_counts / len(self.df) * 100).round(2)

        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Create bar plot
        x = np.arange(len(sentiment_counts))
        bars = plt.bar(x, sentiment_counts.values, 
                      color=['red', 'gray', 'green'])
        
        # Customize plot
        plt.title('Distribution of Tweet Sentiments', pad=20)
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Tweets')
        plt.xticks(x, sentiment_counts.index)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2,
                    height,
                    f'{int(height)}\n({sentiment_percentages[sentiment_counts.index[i]]:.1f}%)',
                    ha='center',
                    va='bottom')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_distribution.png')
        plt.close()

        # Create sarcasm analysis
        sarcasm_by_sentiment = pd.crosstab(self.df['sentiment'], 
                                         self.df['is_sarcastic'])
        
        # Plot sarcasm distribution
        plt.figure(figsize=(10, 6))
        sarcasm_by_sentiment.plot(kind='bar', 
                                stacked=True,
                                color=['lightblue', 'orange'])
        plt.title('Sentiment Distribution with Sarcasm Detection')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Tweets')
        plt.legend(['Non-Sarcastic', 'Sarcastic'])
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_sarcasm_distribution.png')
        plt.close()

        # Save detailed results
        results = {
            'sentiment_counts': sentiment_counts.to_dict(),
            'sentiment_percentages': sentiment_percentages.to_dict(),
            'processing_time': round(time() - start_time, 2),
            'sarcasm_detected': int(self.df['is_sarcastic'].sum()),
            'sarcasm_by_sentiment': sarcasm_by_sentiment.to_dict()
        }

        with open(f'{self.output_dir}/sentiment_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nAnalysis completed in {results['processing_time']} seconds")
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"{sentiment.capitalize()}: {count} tweets ({sentiment_percentages[sentiment]:.1f}%)")
        print(f"\nSarcastic tweets detected: {results['sarcasm_detected']}")

        # Extract and save sarcastic tweets
        sarcastic_tweets = self.df[self.df['is_sarcastic']]['full_text'].values
        with open(f'{self.output_dir}/sarcastic_tweets.txt', 'w', encoding='utf-8') as f:
            for tweet in sarcastic_tweets:
                # Remove any newlines within the tweet and replace with space
                tweet = tweet.replace('\n', ' ').strip()
                f.write(f"{tweet}\n")

        print(f"\nSaved {len(sarcastic_tweets)} sarcastic tweets to 'sarcastic_tweets.txt'")

if __name__ == "__main__":
    analyzer = EfficientSentimentAnalyzer('Election Tweets.json')
    analyzer.analyze_sentiment()
