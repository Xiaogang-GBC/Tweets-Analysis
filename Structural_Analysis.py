
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import networkx as nx
from collections import Counter
import itertools
import os

class StructuralAnalyzer:
    def __init__(self, file_path):
        """Initialize analyzer with data and create output directory"""
        self.output_dir = 'export/structural'
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_data(file_path)
        
    def load_data(self, file_path):
        """Load and preprocess tweet data"""
        print("Loading tweet data...")
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.df = pd.DataFrame(data)
        self.df['created_at'] = pd.to_datetime(self.df['created_at'], 
                                             format='%a %b %d %H:%M:%S +0000 %Y', 
                                             errors='coerce')
        self.df['tweet_length'] = self.df['full_text'].str.len()
        print(f"Loaded {len(self.df)} tweets")

    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in tweets"""
        print("Analyzing temporal patterns...")
        
        # Hourly analysis
        self.df['hour'] = self.df['created_at'].dt.hour
        hourly_counts = self.df['hour'].value_counts().sort_index()
        
        plt.figure(figsize=(12, 6))
        hourly_counts.plot(kind='bar')
        plt.title('Tweet Activity by Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel('Number of Tweets')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/hourly_activity.png')
        plt.close()

        # Daily analysis
        self.df['day_of_week'] = self.df['created_at'].dt.day_name()
        daily_counts = self.df['day_of_week'].value_counts()
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_counts = daily_counts.reindex(days_order)

        plt.figure(figsize=(12, 6))
        daily_counts.plot(kind='bar')
        plt.title('Tweet Activity by Day of Week')
        plt.xlabel('Day')
        plt.ylabel('Number of Tweets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/daily_activity.png')
        plt.close()

    def analyze_tweet_length(self):
        """Analyze distribution of tweet lengths"""
        print("Analyzing tweet lengths...")
        
        plt.figure(figsize=(12, 6))
        sns.histplot(data=self.df, x='tweet_length', bins=50)
        plt.title('Distribution of Tweet Lengths')
        plt.xlabel('Number of Characters')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/tweet_length_distribution.png')
        plt.close()

    def analyze_hashtags(self):
        """Analyze hashtag usage and co-occurrence"""
        print("Analyzing hashtags...")
        
        # Extract hashtags
        hashtag_lists = []
        for tweet in self.df['entities']:
            if isinstance(tweet, dict) and 'hashtags' in tweet:
                hashtags = [tag['text'].lower() for tag in tweet['hashtags']]
                if hashtags:  # Only append if there are hashtags
                    hashtag_lists.append(hashtags)

        # Count individual hashtags
        all_hashtags = [tag for tags in hashtag_lists for tag in tags]
        hashtag_counts = Counter(all_hashtags)

        # Plot top hashtags
        plt.figure(figsize=(12, 6))
        top_hashtags = dict(hashtag_counts.most_common(15))
        plt.bar(top_hashtags.keys(), top_hashtags.values())
        plt.title('Top 15 Hashtags')
        plt.xlabel('Hashtags')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/top_hashtags.png')
        plt.close()

        # Create co-occurrence network
        print("Creating hashtag co-occurrence network...")
        G = nx.Graph()
        edge_weights = {}

        # Count co-occurrences
        for tags in hashtag_lists:
            if len(tags) > 1:
                for pair in itertools.combinations(sorted(tags), 2):
                    if pair in edge_weights:
                        edge_weights[pair] += 1
                    else:
                        edge_weights[pair] = 1

        # Add edges to graph
        for (tag1, tag2), weight in edge_weights.items():
            G.add_edge(tag1, tag2, weight=weight)

        # Draw the network
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)

        # Get node sizes based on degree centrality
        node_sizes = [G.degree(node) * 100 for node in G.nodes()]

        # Get edge weights for line thickness
        edge_weights_list = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]

        # Draw the graph
        nx.draw_networkx(
            G,
            pos,
            node_color='lightgreen',
            node_size=node_sizes,
            width=edge_weights_list,
            edge_color='gray',
            alpha=0.7,
            font_size=8,
            with_labels=True
        )

        plt.title('Hashtag Co-occurrence Network', pad=20)
        plt.axis('off')
        plt.savefig(f'{self.output_dir}/hashtag_cooccurrence_network.png', 
                    bbox_inches='tight',
                    dpi=300)
        plt.close()

        # Save co-occurrence statistics
        cooccurrence_stats = {
            'total_pairs': len(edge_weights),
            'avg_weight': float(np.mean(list(edge_weights.values()))),
            'max_weight': max(edge_weights.values()),
            'top_pairs': {f"{tag1}_{tag2}": weight 
                         for (tag1, tag2), weight in sorted(edge_weights.items(), 
                                                          key=lambda x: x[1], 
                                                          reverse=True)[:10]}
        }

        with open(f'{self.output_dir}/hashtag_cooccurrence_stats.json', 'w') as f:
            json.dump(cooccurrence_stats, f, indent=2)

    def analyze_political_alignment(self):
        """Analyze political alignment distribution"""
        print("Analyzing political alignment...")
        
        political_keywords = {
            'left': ['liberal', 'trudeau', 'ndp', 'singh', 'progressive'],
            'right': ['conservative', 'tory', 'scheer', 'bernier', 'ppc'],
            'center': ['moderate', 'centrist', 'bloc']
        }

        def get_alignment(text):
            if not isinstance(text, str):
                return 'unknown'
            text = text.lower()
            counts = {
                alignment: sum(keyword in text for keyword in keywords)
                for alignment, keywords in political_keywords.items()
            }
            max_count = max(counts.values())
            if max_count == 0:
                return 'neutral'
            max_alignments = [k for k, v in counts.items() if v == max_count]
            return max_alignments[0] if len(max_alignments) == 1 else 'mixed'

        self.df['political_alignment'] = self.df['full_text'].apply(get_alignment)
        alignment_counts = self.df['political_alignment'].value_counts()

        plt.figure(figsize=(10, 6))
        alignment_counts.plot(kind='bar')
        plt.title('Distribution of Political Alignment')
        plt.xlabel('Political Alignment')
        plt.ylabel('Number of Tweets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/political_alignment.png')
        plt.close()

    def analyze_user_network(self):
        """Analyze user interaction network"""
        print("Analyzing user interactions...")
        
        G = nx.Graph()
        for _, tweet in self.df.iterrows():
            try:
                if isinstance(tweet['user'], dict):
                    source = tweet['user'].get('screen_name')
                    if isinstance(tweet.get('entities'), dict):
                        mentions = tweet['entities'].get('user_mentions', [])
                        for mention in mentions:
                            if isinstance(mention, dict):
                                target = mention.get('screen_name')
                                if source and target:
                                    if G.has_edge(source, target):
                                        G[source][target]['weight'] += 1
                                    else:
                                        G.add_edge(source, target, weight=1)
            except:
                continue

        # Filter for visualization
        if len(G.nodes()) > 100:
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:100]
            nodes_to_keep = [node for node, _ in top_nodes]
            G = G.subgraph(nodes_to_keep)

        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(G, k=1, iterations=50)
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw(G, pos,
                node_color='lightblue',
                node_size=100,
                width=edge_weights,
                edge_color='gray',
                alpha=0.7)
        plt.title('User Interaction Network')
        plt.savefig(f'{self.output_dir}/user_network.png', bbox_inches='tight')
        plt.close()

    def run_all_analyses(self):
        """Run all structural analyses"""
        self.analyze_temporal_patterns()
        self.analyze_tweet_length()
        self.analyze_hashtags()
        self.analyze_political_alignment()
        self.analyze_user_network()
        print("Structural analysis complete. Check the 'export/structural' directory for results.")

if __name__ == "__main__":
    analyzer = StructuralAnalyzer('Election Tweets.json')
    analyzer.run_all_analyses()