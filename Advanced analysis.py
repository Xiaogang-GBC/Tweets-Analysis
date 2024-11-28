# install the following packages before running the code
# pip install networkx textblob emoji geopy
# pip install networkx matplotlib seaborn

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import networkx as nx
from collections import Counter
import os
import itertools
from collections import defaultdict

class AdvancedTweetAnalyzer:
    
    def __init__(self, file_path):
        """Initialize with tweet data"""
        self.df = self.load_tweets(file_path)
        self.output_dir = 'advanced_analysis'
        os.makedirs(self.output_dir, exist_ok=True)

    def convert_to_serializable(self, obj):
        if isinstance(obj, dict):
            return {str(key): self.convert_to_serializable(value) 
                    for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return str(obj)
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)
        
    def load_tweets(self, file_path):
        """Load and preprocess tweet data"""
        print("Loading tweet data...")
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        df = pd.DataFrame(data)
        
        # Convert created_at to datetime with explicit format
        df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y', errors='coerce')
        
        # Print data info for debugging
        print("\nDataset Info:")
        print(df.info())
        print("\nSample columns:", df.columns.tolist())
        
        return df
    
    def temporal_analysis(self):
        """Analyze temporal patterns in tweets"""
        print("Analyzing temporal patterns...")
        
        df = self.df.copy()
        
        # Check if created_at is valid
        if 'created_at' not in df.columns or df['created_at'].isnull().all():
            print("Warning: No valid timestamp data found")
            return {
                'hourly_activity': {},
                'daily_activity': {}
            }
        
        # Time-based analysis
        df['hour'] = df['created_at'].dt.hour
        df['day_of_week'] = df['created_at'].dt.day_name()
        df['date'] = df['created_at'].dt.date
        
        # Hourly activity
        hourly_activity = df['hour'].value_counts().sort_index()
        plt.figure(figsize=(12, 6))
        hourly_activity.plot(kind='bar')
        plt.title('Tweet Activity by Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel('Number of Tweets')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/hourly_activity.png')
        plt.close()
        
        # Daily activity
        daily_activity = df['day_of_week'].value_counts()
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_activity = daily_activity.reindex(order)
        plt.figure(figsize=(12, 6))
        daily_activity.plot(kind='bar')
        plt.title('Tweet Activity by Day of Week')
        plt.xlabel('Day')
        plt.ylabel('Number of Tweets')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/daily_activity.png')
        plt.close()
        
        return {
            'hourly_activity': hourly_activity.to_dict(),
            'daily_activity': daily_activity.to_dict()
        }

    def user_influence_analysis(self):
        """Analyze user influence and engagement"""
        print("Analyzing user influence...")
        
        # Extract user metrics with error handling
        user_metrics_list = []
        for _, tweet in self.df.iterrows():
            try:
                if isinstance(tweet['user'], dict):
                    metrics = {
                        'user': tweet['user'].get('screen_name', 'unknown'),
                        'followers': tweet['user'].get('followers_count', 0),
                        'friends': tweet['user'].get('friends_count', 0),
                        'statuses': tweet['user'].get('statuses_count', 0),
                        'engagement': (tweet.get('favorite_count', 0) or 0) + 
                                    (tweet.get('retweet_count', 0) or 0)
                    }
                    user_metrics_list.append(metrics)
            except (TypeError, AttributeError, KeyError) as e:
                print(f"Error processing tweet: {e}")
                continue
        
        if not user_metrics_list:
            print("Warning: No valid user metrics found")
            return {
                'top_influential_users': [],
                'avg_engagement_rate': 0
            }
        
        user_metrics = pd.DataFrame(user_metrics_list)
        
        # Calculate influence scores
        user_metrics['influence_score'] = (
            np.log1p(user_metrics['followers']) * 0.5 +
            np.log1p(user_metrics['engagement']) * 0.3 +
            np.log1p(user_metrics['statuses']) * 0.2
        )
        
        # Top influential users
        top_users = user_metrics.nlargest(20, 'influence_score')
        
        # Create visualization only if we have valid data
        if not user_metrics.empty:
            plt.figure(figsize=(12, 6))
            sns.scatterplot(data=user_metrics, x='followers', y='engagement', alpha=0.5)
            plt.title('User Influence: Followers vs Engagement')
            plt.xlabel('Number of Followers (log scale)')
            plt.ylabel('Engagement (log scale)')
            plt.xscale('log')
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/user_influence.png')
            plt.close()
        
        return {
            'top_influential_users': top_users.to_dict('records'),
            'avg_engagement_rate': user_metrics['engagement'].mean()
        }

    def content_analysis(self):
        """Analyze tweet content patterns"""
        print("Analyzing content patterns...")
        
        # Calculate tweet lengths with error handling
        self.df['tweet_length'] = self.df['full_text'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
        
        # Analyze media types
        def get_media_type(tweet):
            try:
                if isinstance(tweet.get('extended_entities'), dict):
                    if 'media' in tweet['extended_entities']:
                        return tweet['extended_entities']['media'][0]['type']
                return 'text_only'
            except (TypeError, KeyError):
                return 'text_only'
        
        self.df['media_type'] = self.df.apply(get_media_type, axis=1)
        
        # Results
        content_stats = {
            'avg_tweet_length': self.df['tweet_length'].mean(),
            'media_distribution': self.df['media_type'].value_counts().to_dict()
        }
        
        # Visualize tweet length distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='tweet_length', bins=50)
        plt.title('Distribution of Tweet Lengths')
        plt.xlabel('Tweet Length (characters)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/tweet_length_distribution.png')
        plt.close()
        
        return content_stats
    def interaction_network_analysis(self):
        """Analyze user interaction networks"""
        print("Analyzing interaction networks...")
        
        # Initialize graph and containers
        G = nx.Graph()
        edges = []
        metrics = {
            'total_interactions': 0,
            'unique_users': 0,
            'density': 0,
            'avg_clustering': 0,
            'most_interactive_users': []
        }
        
        try:
            # Add edges for mentions and replies
            for _, tweet in self.df.iterrows():
                try:
                    if isinstance(tweet.get('user'), dict):
                        source = tweet['user'].get('screen_name', None)
                        if not source:
                            continue
                        
                        # Add mentions
                        if isinstance(tweet.get('entities'), dict):
                            mentions = tweet['entities'].get('user_mentions', [])
                            for mention in mentions:
                                if isinstance(mention, dict):
                                    target = mention.get('screen_name')
                                    if target:
                                        edges.append((source, target))
                except (AttributeError, KeyError, TypeError) as e:
                    continue
            
            if not edges:
                print("No interactions found in the dataset")
                return metrics
            
            # Add edges to graph with weights
            edge_weights = Counter(edges)
            for (source, target), weight in edge_weights.items():
                G.add_edge(source, target, weight=weight)
            
            if len(G.nodes()) == 0:
                print("No valid nodes found in the network")
                return metrics
            
            # Calculate network metrics
            metrics.update({
                'total_interactions': len(edges),
                'unique_users': len(G.nodes()),
                'density': nx.density(G)
            })
            
            try:
                metrics['avg_clustering'] = nx.average_clustering(G)
            except Exception as e:
                print(f"Error calculating clustering coefficient: {e}")
                metrics['avg_clustering'] = 0
            
            # Find most active users in interactions
            try:
                degree_centrality = nx.degree_centrality(G)
                top_users = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                metrics['most_interactive_users'] = [
                    {'user': user, 'centrality': round(cent, 4)} 
                    for user, cent in top_users
                ]
            except Exception as e:
                print(f"Error calculating user centrality: {e}")
                metrics['most_interactive_users'] = []
            
            # Visualize network (limited to top interactions for visibility)
            try:
                if len(G.nodes()) > 0:
                    plt.figure(figsize=(12, 12))
                    
                    # Create layout with error handling
                    try:
                        pos = nx.spring_layout(G, k=1, iterations=50)
                    except Exception as e:
                        print(f"Error in spring layout, trying alternative layout: {e}")
                        pos = nx.circular_layout(G)
                    
                    # Draw network
                    nx.draw(G, pos, 
                        node_color='lightblue',
                        node_size=100,
                        alpha=0.6,
                        with_labels=False)
                    
                    # Add labels for top users only if available
                    if metrics['most_interactive_users']:
                        top_user_labels = {
                            user_data['user']: user_data['user'] 
                            for user_data in metrics['most_interactive_users']
                        }
                        nx.draw_networkx_labels(G, pos, labels=top_user_labels)
                    
                    plt.title('User Interaction Network')
                    plt.savefig(f'{self.output_dir}/interaction_network.png', 
                            bbox_inches='tight', 
                            pad_inches=0.1)
                    plt.close()
            except Exception as e:
                print(f"Error creating network visualization: {e}")
            
        except Exception as e:
            print(f"Error in interaction network analysis: {e}")
            import traceback
            traceback.print_exc()
        
        return metrics

    def hashtag_cooccurrence_analysis(self):
        """Analyze hashtag co-occurrence patterns"""
        print("Analyzing hashtag patterns...")
        
        hashtag_pairs = []
        hashtag_counts = Counter()
        
        # Collect hashtags and their co-occurrences
        for _, tweet in self.df.iterrows():
            try:
                if isinstance(tweet.get('entities'), dict):
                    hashtags = [tag['text'].lower() 
                            for tag in tweet['entities'].get('hashtags', [])
                            if isinstance(tag, dict) and 'text' in tag]
                    
                    # Count individual hashtags
                    hashtag_counts.update(hashtags)
                    
                    # Count co-occurrences
                    if len(hashtags) > 1:
                        pairs = list(itertools.combinations(sorted(hashtags), 2))
                        hashtag_pairs.extend(pairs)
            except (AttributeError, KeyError, TypeError) as e:
                continue
        
        # Count co-occurrences and convert to string keys
        cooccurrence = Counter(hashtag_pairs)
        cooccurrence_dict = {f"{tag1}_{tag2}": count 
                            for (tag1, tag2), count in cooccurrence.most_common(20)}
        
        # Create visualization of top hashtags
        plt.figure(figsize=(12, 6))
        most_common = hashtag_counts.most_common(15)
        if most_common:
            tags, counts = zip(*most_common)
            plt.bar(range(len(counts)), counts)
            plt.xticks(range(len(counts)), tags, rotation=45, ha='right')
            plt.title('Most Common Hashtags')
            plt.xlabel('Hashtags')
            plt.ylabel('Frequency')
            plt.tight_layout()
        plt.savefig(f'{self.output_dir}/top_hashtags.png')
        plt.close()
        
        # Create network visualization for hashtag co-occurrence
        if cooccurrence:
            G = nx.Graph()
            
            # Add edges for top co-occurrences
            for (tag1, tag2), weight in cooccurrence.most_common(50):
                G.add_edge(tag1, tag2, weight=weight)
            
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw network
            nx.draw(G, pos,
                node_color='lightgreen',
                node_size=1000,
                alpha=0.6,
                with_labels=True,
                font_size=8)
            
            plt.title('Hashtag Co-occurrence Network')
            plt.savefig(f'{self.output_dir}/hashtag_network.png', 
                    bbox_inches='tight', 
                    pad_inches=0.1)
            plt.close()
        
        return {
            'top_hashtags': dict(hashtag_counts.most_common(20)),
            'top_cooccurrences': cooccurrence_dict
        }

    def political_alignment_analysis(self):
        """Analyze political alignment and polarization"""
        print("Analyzing political alignment...")
        
        # Define political keywords (expanded for Canadian context)
        political_keywords = {
            'left': [
                'liberal', 'trudeau', 'ndp', 'singh', 'progressive',
                'left', 'progressive', 'green party', 'elizabeth may',
                'socialist', 'labour', 'union'
            ],
            'right': [
                'conservative', 'tory', 'scheer', 'bernier', 'ppc',
                'right', 'reform', 'populist', 'peoples party'
            ],
            'center': [
                'moderate', 'centrist', 'bloc', 'bloc québécois',
                'blanchet', 'independent'
            ]
        }
        
        def get_political_alignment(text):
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
            
            # Check for multiple equal max values
            max_alignments = [k for k, v in counts.items() if v == max_count]
            if len(max_alignments) > 1:
                return 'mixed'
            
            return max_alignments[0]
        
        # Analyze tweets for political alignment
        self.df['political_alignment'] = self.df['full_text'].apply(get_political_alignment)
        
        # Calculate distributions
        alignment_dist = self.df['political_alignment'].value_counts()
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        alignment_dist.plot(kind='bar')
        plt.title('Distribution of Political Alignment in Tweets')
        plt.xlabel('Political Alignment')
        plt.ylabel('Number of Tweets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/political_alignment.png')
        plt.close()
        
        # Calculate polarization metrics
        total_political = len(self.df[self.df['political_alignment'].isin(['left', 'right'])])
        if total_political > 0:
            left_count = len(self.df[self.df['political_alignment'] == 'left'])
            right_count = len(self.df[self.df['political_alignment'] == 'right'])
            polarization_index = abs(left_count - right_count) / total_political
        else:
            polarization_index = 0
        
        # Analyze hashtag usage by political alignment
        hashtags_by_alignment = defaultdict(Counter)
        for _, tweet in self.df.iterrows():
            try:
                alignment = tweet['political_alignment']
                if isinstance(tweet.get('entities'), dict):
                    hashtags = [tag['text'].lower() 
                              for tag in tweet['entities'].get('hashtags', [])
                              if isinstance(tag, dict) and 'text' in tag]
                    hashtags_by_alignment[alignment].update(hashtags)
            except (AttributeError, KeyError, TypeError):
                continue
        
        return {
            'distribution': alignment_dist.to_dict(),
            'polarization_index': polarization_index,
            'top_hashtags_by_alignment': {
                alignment: dict(counter.most_common(10))
                for alignment, counter in hashtags_by_alignment.items()
            }
        }

    def run_full_analysis(self):
        """Run all analyses and compile results"""
        try:
            results = {
                'temporal_patterns': self.temporal_analysis(),
                'user_influence': self.user_influence_analysis(),
                'content_patterns': self.content_analysis(),
                'interaction_network': self.interaction_network_analysis(),
                'hashtag_patterns': self.hashtag_cooccurrence_analysis(),
                'political_analysis': self.political_alignment_analysis()
            }
            
            # Convert results to JSON-serializable format
            serializable_results = self.convert_to_serializable(results)
            
            # Save results
            with open(f'{self.output_dir}/full_analysis_results.json', 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print("\nAnalysis complete! Results saved in 'advanced_analysis' directory:")
            print("- temporal_patterns (hourly_activity.png, daily_activity.png)")
            print("- user_influence (user_influence.png)")
            print("- content_patterns (tweet_length_distribution.png)")
            print("- interaction_network (interaction_network.png)")
            print("- hashtag_patterns (top_hashtags.png, hashtag_network.png)")
            print("- political_analysis (political_alignment.png)")
            print("- full_analysis_results.json")
            
            return serializable_results
        
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
        
    def get_user_screen_name(self, user_data):
        """Safely extract screen name from user data"""
        if isinstance(user_data, dict):
            return user_data.get('screen_name', '')
        return ''

    def get_mentions(self, tweet):
        """Safely extract mentions from tweet"""
        try:
            if isinstance(tweet.get('entities'), dict):
                mentions = tweet['entities'].get('user_mentions', [])
                return [mention['screen_name'] for mention in mentions 
                    if isinstance(mention, dict) and 'screen_name' in mention]
        except Exception:
            pass
        return []

    def filter_graph_for_visualization(self, G, max_nodes=100):
        """Filter large graphs for better visualization"""
        if len(G.nodes()) > max_nodes:
            # Get top nodes by degree
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            top_node_ids = [node for node, _ in top_nodes]
            
            # Create subgraph with only top nodes
            H = G.subgraph(top_node_ids)
            return H
        return G
    

if __name__ == "__main__":
    # Add debugging information
    print("Starting analysis...")
    try:
        analyzer = AdvancedTweetAnalyzer('Election Tweets.json')
        results = analyzer.run_full_analysis()
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()