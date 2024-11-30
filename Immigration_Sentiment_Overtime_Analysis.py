
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import numpy as np
import os

class CandidateImmigrationAnalyzer:
    def __init__(self, file_path):
        self.output_dir = 'export/candidate_analysis'
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_data(file_path)
        
    def load_data(self, file_path):
        print("Loading tweet data...")
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.df = pd.DataFrame(data)
        self.df['created_at'] = pd.to_datetime(self.df['created_at'])
        print(f"Loaded {len(self.df)} tweets")

    def identify_immigration_tweets(self, text):
        """Identify immigration-related tweets"""
        immigration_keywords = [
            'immigration', 'immigrant', 'refugee', 'asylum', 
            'border', 'migrate', 'citizenship', 'newcomer'
        ]
        text = str(text).lower()
        return any(keyword in text for keyword in immigration_keywords)

    def analyze_candidate_mentions(self):
        """Analyze mentions of candidates in immigration context"""
        print("Analyzing candidate mentions in immigration context...")
        
        # Add immigration flag
        self.df['is_immigration'] = self.df['full_text'].apply(self.identify_immigration_tweets)
        
        # Add candidate flags
        self.df['mentions_trudeau'] = self.df['full_text'].str.lower().str.contains('trudeau')
        self.df['mentions_scheer'] = self.df['full_text'].str.lower().str.contains('scheer')
        
        # Create daily timelines
        daily_data = self.df.resample('D', on='created_at').agg({
            'is_immigration': 'sum',
            'mentions_trudeau': 'sum',
            'mentions_scheer': 'sum'
        }).fillna(0)
        
        # Calculate immigration mentions for each candidate
        trudeau_immigration = self.df[
            self.df['mentions_trudeau'] & self.df['is_immigration']
        ].resample('D', on='created_at').size()
        
        scheer_immigration = self.df[
            self.df['mentions_scheer'] & self.df['is_immigration']
        ].resample('D', on='created_at').size()
        
        # Create timeline visualization
        plt.figure(figsize=(15, 8))
        plt.plot(trudeau_immigration.index, 
                trudeau_immigration.values, 
                label='Trudeau', 
                color='red',
                marker='o')
        plt.plot(scheer_immigration.index, 
                scheer_immigration.values, 
                label='Scheer', 
                color='blue',
                marker='o')
        
        plt.title('Candidate Mentions in Immigration-Related Tweets Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Mentions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/candidate_immigration_timeline.png')
        plt.close()

        # Analyze sentiment in immigration tweets
        def get_sentiment(text):
            text = str(text).lower()
            positive_words = {'support', 'welcome', 'positive', 'good', 'great', 'excellent'}
            negative_words = {'against', 'bad', 'worse', 'terrible', 'problem', 'crisis'}
            
            pos_count = sum(word in text for word in positive_words)
            neg_count = sum(word in text for word in negative_words)
            
            if pos_count > neg_count:
                return 'Positive'
            elif neg_count > pos_count:
                return 'Negative'
            return 'Neutral'

        # Add sentiment analysis
        immigration_tweets = self.df[self.df['is_immigration']].copy()
        immigration_tweets['sentiment'] = immigration_tweets['full_text'].apply(get_sentiment)

        # Analyze sentiment distribution for each candidate
        trudeau_sentiments = immigration_tweets[
            immigration_tweets['mentions_trudeau']
        ]['sentiment'].value_counts()
        
        scheer_sentiments = immigration_tweets[
            immigration_tweets['mentions_scheer']
        ]['sentiment'].value_counts()

        # Create sentiment comparison visualization
        plt.figure(figsize=(12, 6))
        width = 0.35
        x = np.arange(3)  # Three sentiment categories
        
        plt.bar(x - width/2, 
                [trudeau_sentiments.get('Positive', 0),
                 trudeau_sentiments.get('Neutral', 0),
                 trudeau_sentiments.get('Negative', 0)],
                width,
                label='Trudeau',
                color='red',
                alpha=0.7)
        
        plt.bar(x + width/2, 
                [scheer_sentiments.get('Positive', 0),
                 scheer_sentiments.get('Neutral', 0),
                 scheer_sentiments.get('Negative', 0)],
                width,
                label='Scheer',
                color='blue',
                alpha=0.7)

        plt.xlabel('Sentiment')
        plt.ylabel('Number of Tweets')
        plt.title('Sentiment Distribution in Immigration-Related Tweets')
        plt.xticks(x, ['Positive', 'Neutral', 'Negative'])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/candidate_immigration_sentiment.png')
        plt.close()

        # Save statistics
        stats = {
            'total_immigration_tweets': int(self.df['is_immigration'].sum()),
            'trudeau_immigration_mentions': int(trudeau_immigration.sum()),
            'scheer_immigration_mentions': int(scheer_immigration.sum()),
            'trudeau_sentiment': trudeau_sentiments.to_dict(),
            'scheer_sentiment': scheer_sentiments.to_dict(),
            'peak_days': {
                'trudeau': trudeau_immigration.idxmax().strftime('%Y-%m-%d'),
                'scheer': scheer_immigration.idxmax().strftime('%Y-%m-%d')
            }
        }

        with open(f'{self.output_dir}/candidate_immigration_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        # Print summary
        print("\nAnalysis Summary:")
        print(f"Total immigration-related tweets: {stats['total_immigration_tweets']}")
        print(f"Trudeau mentions: {stats['trudeau_immigration_mentions']}")
        print(f"Scheer mentions: {stats['scheer_immigration_mentions']}")
        print(f"Trudeau peak day: {stats['peak_days']['trudeau']}")
        print(f"Scheer peak day: {stats['peak_days']['scheer']}")

if __name__ == "__main__":
    analyzer = CandidateImmigrationAnalyzer('Election Tweets.json')
    analyzer.analyze_candidate_mentions()
