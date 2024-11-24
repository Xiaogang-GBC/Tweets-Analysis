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
import os

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    print("NLTK data already downloaded or error in downloading")

def load_tweets(file_path):
    """Load tweets from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return pd.DataFrame()

def preprocess_text(text):
    """Preprocess tweet text"""
    # Convert to string if not already
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags symbols but keep the text
    text = re.sub(r'#', '', text)
    
    # Remove RT (retweet) marks
    text = re.sub(r'^RT[\s]+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def create_wordcloud(text_data, title, output_path, 
                    width=1600, height=800, 
                    background_color='white',
                    custom_stopwords=None):
    """Create and save word cloud"""
    # Get standard English stop words
    stop_words = set(stopwords.words('english'))
    
    # Add custom stop words
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    
    # Create word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        stopwords=stop_words,
        min_font_size=10,
        max_font_size=150,
        max_words=200,
        collocations=False
    ).generate(text_data)
    
    # Create figure
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=20, pad=20)
    
    # Save figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Return word frequencies for additional analysis
    words = word_tokenize(text_data)
    word_freq = Counter(word for word in words if word.lower() not in stop_words)
    return word_freq

def analyze_tweets_wordcloud():
    # Create output directory
    output_dir = 'wordcloud_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading tweets...")
    df = load_tweets('Election Tweets.json')
    
    if df.empty:
        print("Error: No data loaded")
        return
    
    # Custom stop words relevant to election tweets
    custom_stopwords = {
        'amp', 'rt', 'twitter', 'tweet', 'tweets', 'https', 'co', 
        'would', 'could', 'should', 'said', 'says', 'one', 'will',
        'time', 'today', 'day', 'now', 'get', 'got', 'going'
    }
    
    print("Processing tweets...")
    # Combine all tweets and preprocess
    all_tweets_text = ' '.join(df['full_text'].apply(preprocess_text))
    
    print("Creating general word cloud...")
    # Create general word cloud
    word_freq = create_wordcloud(
        all_tweets_text,
        'Word Cloud of Election Tweets',
        f'{output_dir}/general_wordcloud.png',
        custom_stopwords=custom_stopwords
    )
    
    # Create word clouds by sentiment if sentiment data exists
    if 'sentiment' in df.columns:
        sentiment_labels = {
            0: 'Negative',
            1: 'Neutral',
            2: 'Positive'
        }
        
        for sentiment, label in sentiment_labels.items():
            sentiment_tweets = df[df['sentiment'] == sentiment]
            if len(sentiment_tweets) > 0:
                sentiment_text = ' '.join(sentiment_tweets['full_text'].apply(preprocess_text))
                print(f"Creating word cloud for {label} sentiment...")
                create_wordcloud(
                    sentiment_text,
                    f'Word Cloud of {label} Election Tweets',
                    f'{output_dir}/{label.lower()}_wordcloud.png',
                    custom_stopwords=custom_stopwords
                )
    
    # Save word frequency data
    print("Saving word frequency data...")
    top_words = pd.DataFrame(
        word_freq.most_common(50),
        columns=['Word', 'Frequency']
    )
    
    # Create bar plot of top words
    plt.figure(figsize=(15, 8))
    bars = plt.bar(
        range(20),
        top_words['Frequency'][:20],
        color='skyblue'
    )
    plt.xticks(
        range(20),
        top_words['Word'][:20],
        rotation=45,
        ha='right'
    )
    plt.title('Top 20 Most Frequent Words in Election Tweets')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    
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
    plt.savefig(f'{output_dir}/top_words_barplot.png')
    plt.close()
    
    # Save frequency data to CSV
    top_words.to_csv(f'{output_dir}/word_frequencies.csv', index=False)
    
    # Save basic statistics
    stats = {
        'total_tweets': len(df),
        'unique_words': len(word_freq),
        'total_words': sum(word_freq.values()),
        'avg_words_per_tweet': sum(word_freq.values()) / len(df)
    }
    
    with open(f'{output_dir}/word_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\nWord cloud analysis complete! Files saved in 'wordcloud_analysis' directory:")
    print("- general_wordcloud.png")
    print("- top_words_barplot.png")
    print("- word_frequencies.csv")
    print("- word_statistics.json")
    if 'sentiment' in df.columns:
        print("- negative_wordcloud.png")
        print("- neutral_wordcloud.png")
        print("- positive_wordcloud.png")

if __name__ == "__main__":
    analyze_tweets_wordcloud()