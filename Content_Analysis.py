
import json
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import os

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
except:
    print("NLTK data already downloaded or error in downloading")

class ContentAnalyzer:
    def __init__(self, file_path):
        """Initialize analyzer with data and create output directory"""
        self.output_dir = 'export/content'
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_data(file_path)
        self.stop_words = set(stopwords.words('english'))
        self.custom_stop_words = {
            'amp', 'rt', 'twitter', 'tweet', 'tweets', 'https', 'co', 
            'would', 'could', 'should', 'said', 'says', 'one', 'will',
            'time', 'today', 'day', 'now', 'get', 'got', 'going'
        }
        self.stop_words.update(self.custom_stop_words)

    def load_data(self, file_path):
        """Load and preprocess tweet data"""
        print("Loading tweet data...")
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.df = pd.DataFrame(data)
        print(f"Loaded {len(self.df)} tweets")

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags completely (not just the # symbol)
        text = re.sub(r'#\w+', '', text)
        
        # Remove RT marks
        text = re.sub(r'^RT[\s]+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text

    def create_word_cloud(self):
        """Generate and save word cloud"""
        print("Creating word cloud...")
        
        # Combine all preprocessed tweets
        all_tweets_text = ' '.join(self.df['full_text'].apply(self.preprocess_text))
        
        # Create word cloud
        wordcloud = WordCloud(
            width=1600,
            height=800,
            background_color='white',
            stopwords=self.stop_words,
            min_font_size=10,
            max_font_size=150,
            max_words=200,
            collocations=False
        ).generate(all_tweets_text)
        
        # Save word cloud
        plt.figure(figsize=(20, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Election Tweets', fontsize=20, pad=20)
        plt.savefig(f'{self.output_dir}/wordcloud.png', dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_word_frequency(self):
        """Analyze and visualize word frequencies"""
        print("Analyzing word frequencies...")
        
        # Get all words from preprocessed tweets
        all_words = []
        for text in self.df['full_text']:
            processed_text = self.preprocess_text(text)
            words = word_tokenize(processed_text)
            all_words.extend([word for word in words 
                            if word.lower() not in self.stop_words
                            and len(word) > 2])  # Filter out very short words
        
        # Count word frequencies
        word_freq = Counter(all_words)
        
        # Create visualization of top 20 words
        top_words = pd.DataFrame(word_freq.most_common(20), 
                               columns=['Word', 'Frequency'])
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(
            range(len(top_words)),
            top_words['Frequency'],
            color='skyblue'
        )
        plt.title('Top 20 Most Frequent Words in Election Tweets')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.xticks(
            range(len(top_words)),
            top_words['Word'],
            rotation=45,
            ha='right'
        )
        
        # Add frequency labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom'
            )
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/top_words.png')
        plt.close()
        
        # Save frequency data
        top_words.to_csv(f'{self.output_dir}/word_frequencies.csv', index=False)

    def analyze_word_pairs(self):
        """Analyze common word pairs (bigrams)"""
        print("Analyzing word pairs...")
        
        # Get bigrams from preprocessed tweets
        bigram_list = []
        for text in self.df['full_text']:
            processed_text = self.preprocess_text(text)
            words = word_tokenize(processed_text)
            # Filter stop words and short words
            words = [word for word in words 
                    if word.lower() not in self.stop_words
                    and len(word) > 2]
            # Generate bigrams
            tweet_bigrams = list(ngrams(words, 2))
            bigram_list.extend(tweet_bigrams)
        
        # Count bigram frequencies
        bigram_freq = Counter(bigram_list)
        
        # Create visualization of top 15 bigrams
        top_bigrams = pd.DataFrame(bigram_freq.most_common(15), 
                                 columns=['Bigram', 'Frequency'])
        top_bigrams['Bigram'] = top_bigrams['Bigram'].apply(lambda x: ' '.join(x))
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(
            range(len(top_bigrams)),
            top_bigrams['Frequency'],
            color='lightgreen'
        )
        plt.title('Top 15 Most Frequent Word Pairs in Election Tweets')
        plt.xlabel('Word Pairs')
        plt.ylabel('Frequency')
        plt.xticks(
            range(len(top_bigrams)),
            top_bigrams['Bigram'],
            rotation=45,
            ha='right'
        )
        
        # Add frequency labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom'
            )
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/top_word_pairs.png')
        plt.close()
        
        # Save bigram data
        top_bigrams.to_csv(f'{self.output_dir}/word_pair_frequencies.csv', index=False)

    def run_all_analyses(self):
        """Run all content analyses"""
        self.create_word_cloud()
        self.analyze_word_frequency()
        self.analyze_word_pairs()
        print("Content analysis complete. Check the 'export/content' directory for results.")

if __name__ == "__main__":
    analyzer = ContentAnalyzer('Election Tweets.json')
    analyzer.run_all_analyses()
