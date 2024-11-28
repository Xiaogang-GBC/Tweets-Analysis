
# main_analysis.py
# Required packages: networkx wordcloud nltk pandas numpy matplotlib seaborn

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import networkx as nx
from collections import Counter, defaultdict
import os
import itertools
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# Download NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    print("NLTK data already downloaded or error in downloading")

class TwitterAnalyzer:
    def __init__(self, file_path):
        """Initialize analyzer with data and create output directory"""
        self.output_dir = 'export'
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = self.load_data(file_path)

    def load_data(self, file_path):
        """Load and preprocess tweet data"""
        print("Loading tweet data...")
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        df = pd.DataFrame(data)
        df['created_at'] = pd.to_datetime(df['created_at'], 
                                        format='%a %b %d %H:%M:%S +0000 %Y', 
                                        errors='coerce')
        return df

    def run_analysis(self):
        """Run all analyses except sentiment"""
        print("\nStarting comprehensive analysis...")
        
        # 1. Network Analysis
        print("\nPerforming network analysis...")
        self.perform_network_analysis()
        
        # 2. Word Cloud Analysis
        print("\nGenerating word clouds...")
        self.perform_wordcloud_analysis()
        
        # 3. Additional Analyses
        print("\nPerforming additional analyses...")
        self.perform_additional_analyses()
        
        print("\nAnalysis complete! Results saved in 'export' directory")

    def perform_network_analysis(self):
        """Network and interaction analysis"""
        # Create interaction network
        G = nx.Graph()
        edges = []
        
        for _, tweet in self.df.iterrows():
            try:
                if isinstance(tweet.get('user'), dict):
                    source = tweet['user'].get('screen_name')
                    if isinstance(tweet.get('entities'), dict):
                        mentions = tweet['entities'].get('user_mentions', [])
                        for mention in mentions:
                            if isinstance(mention, dict):
                                target = mention.get('screen_name')
                                if source and target:
                                    edges.append((source, target))
            except Exception:
                continue

        # Add edges and create visualization
        edge_weights = Counter(edges)
        for (source, target), weight in edge_weights.items():
            G.add_edge(source, target, weight=weight)

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=1, iterations=50)
        nx.draw(G, pos, node_color='lightblue', node_size=100, alpha=0.6)
        plt.title('User Interaction Network')
        plt.savefig(f'{self.output_dir}/interaction_network.png', bbox_inches='tight')
        plt.close()

        # Save network statistics
        stats = {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes(),
            'density': nx.density(G)
        }
        with open(f'{self.output_dir}/network_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

    def perform_wordcloud_analysis(self):
        """Generate word clouds and frequency analysis"""
        # Preprocess text
        all_tweets_text = ' '.join(self.df['full_text'].apply(self.preprocess_text))
        
        # Create word cloud
        stop_words = set(stopwords.words('english'))
        custom_stopwords = {
            'amp', 'rt', 'twitter', 'tweet', 'tweets', 'https', 'co', 
            'would', 'could', 'should', 'said', 'says'
        }
        stop_words.update(custom_stopwords)

        wordcloud = WordCloud(
            width=1600, height=800,
            background_color='white',
            stopwords=stop_words,
            max_words=200
        ).generate(all_tweets_text)

        # Save word cloud
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Election Tweets', fontsize=20, pad=20)
        plt.savefig(f'{self.output_dir}/wordcloud.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Generate word frequency analysis
        words = word_tokenize(all_tweets_text)
        word_freq = Counter(word for word in words if word.lower() not in stop_words)
        
        # Save top words
        top_words = pd.DataFrame(word_freq.most_common(50), 
                               columns=['Word', 'Frequency'])
        top_words.to_csv(f'{self.output_dir}/word_frequencies.csv', index=False)

    def perform_additional_analyses(self):
        """Perform temporal and engagement analyses"""
        # Temporal analysis
        self.df['hour'] = self.df['created_at'].dt.hour
        self.df['day'] = self.df['created_at'].dt.day_name()
        
        # Hourly activity
        hourly_activity = self.df['hour'].value_counts().sort_index()
        plt.figure(figsize=(12, 6))
        hourly_activity.plot(kind='bar')
        plt.title('Tweet Activity by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Tweets')
        plt.savefig(f'{self.output_dir}/hourly_activity.png')
        plt.close()

        # Daily activity
        daily_activity = self.df['day'].value_counts()
        plt.figure(figsize=(12, 6))
        daily_activity.plot(kind='bar')
        plt.title('Tweet Activity by Day')
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Tweets')
        plt.savefig(f'{self.output_dir}/daily_activity.png')
        plt.close()

        # Engagement analysis
        engagement_data = self.df.apply(
            lambda x: (x.get('favorite_count', 0) or 0) + 
                     (x.get('retweet_count', 0) or 0),
            axis=1
        )
        
        plt.figure(figsize=(10, 6))
        plt.hist(np.log1p(engagement_data), bins=50)
        plt.title('Distribution of Tweet Engagement (log scale)')
        plt.xlabel('Log Engagement (Favorites + Retweets)')
        plt.ylabel('Count')
        plt.savefig(f'{self.output_dir}/engagement_distribution.png')
        plt.close()

        # Save activity statistics
        activity_stats = {
            'hourly_activity': hourly_activity.to_dict(),
            'daily_activity': daily_activity.to_dict(),
            'total_engagement': int(engagement_data.sum()),
            'average_engagement': float(engagement_data.mean()),
            'median_engagement': float(engagement_data.median())
        }
        with open(f'{self.output_dir}/activity_stats.json', 'w') as f:
            json.dump(activity_stats, f, indent=2)

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'^RT[\s]+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

if __name__ == "__main__":
    try:
        analyzer = TwitterAnalyzer('Election Tweets.json')
        analyzer.run_analysis()
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
