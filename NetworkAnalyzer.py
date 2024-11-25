
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import community.community_louvain as community_louvain
import warnings
warnings.filterwarnings('ignore')

class TwitterNetworkAnalyzer:
    def __init__(self, file_path, min_influence_threshold=0.01):
        """
        Initialize the analyzer with tweet data
        min_influence_threshold: minimum influence score to include in visualization (0-1)
        """
        self.min_influence_threshold = min_influence_threshold
        self.load_data(file_path)
        self.G = nx.Graph()
        
    def load_data(self, file_path):
        """Load tweet data from JSON file"""
        print("Loading tweet data...")
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.df = pd.DataFrame(data)
        print(f"Loaded {len(self.df)} tweets")

    def create_filtered_network(self):
        """Create network from tweet interactions with filtering"""
        print("Creating filtered network from interactions...")
        
        # Track edges and weights
        edge_weights = {}
        node_weights = {}  # Track node importance
        
        for _, tweet in self.df.iterrows():
            try:
                if isinstance(tweet.get('user'), dict):
                    source = tweet['user'].get('screen_name')
                    followers = tweet['user'].get('followers_count', 0)
                    
                    # Update node importance
                    if source:
                        if source not in node_weights:
                            node_weights[source] = {
                                'followers': followers,
                                'mentions': 0,
                                'engagement': tweet.get('favorite_count', 0) + tweet.get('retweet_count', 0)
                            }
                        else:
                            node_weights[source]['engagement'] += tweet.get('favorite_count', 0) + tweet.get('retweet_count', 0)
                    
                    # Add mentions
                    if isinstance(tweet.get('entities'), dict):
                        mentions = tweet['entities'].get('user_mentions', [])
                        for mention in mentions:
                            if isinstance(mention, dict):
                                target = mention.get('screen_name')
                                if source and target:
                                    edge = tuple(sorted([source, target]))
                                    edge_weights[edge] = edge_weights.get(edge, 0) + 1
                                    
                                    # Update mention count
                                    if target not in node_weights:
                                        node_weights[target] = {'followers': 0, 'mentions': 1, 'engagement': 0}
                                    else:
                                        node_weights[target]['mentions'] += 1
            
            except Exception as e:
                continue
        
        # Calculate influence scores
        max_followers = max([d['followers'] for d in node_weights.values()]) if node_weights else 1
        max_mentions = max([d['mentions'] for d in node_weights.values()]) if node_weights else 1
        max_engagement = max([d['engagement'] for d in node_weights.values()]) if node_weights else 1
        
        influence_scores = {}
        for node, weights in node_weights.items():
            # Normalize each component
            norm_followers = weights['followers'] / max_followers if max_followers > 0 else 0
            norm_mentions = weights['mentions'] / max_mentions if max_mentions > 0 else 0
            norm_engagement = weights['engagement'] / max_engagement if max_engagement > 0 else 0
            
            # Calculate weighted influence score
            influence_scores[node] = (
                0.4 * norm_followers +  # Follower count importance
                0.3 * norm_mentions +   # Mention frequency importance
                0.3 * norm_engagement   # Engagement importance
            )
        
        # Filter nodes based on influence threshold
        influential_nodes = {node for node, score in influence_scores.items() 
                           if score >= self.min_influence_threshold}
        
        # Create filtered graph
        self.G = nx.Graph()
        
        # Add filtered edges
        for (source, target), weight in edge_weights.items():
            if source in influential_nodes and target in influential_nodes:
                self.G.add_edge(source, target, weight=weight)
        
        # Store influence scores for visualization
        self.influence_scores = {node: score for node, score in influence_scores.items()
                               if node in self.G.nodes()}
        
        print(f"Created filtered network with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        return influence_scores

    def visualize_network(self, communities, output_path='network_graph.png'):
        """Create and save network visualization with improved layout"""
        print("Creating network visualization...")
        
        if len(self.G.nodes()) == 0:
            print("No nodes to visualize after filtering")
            return
        
        # Set up the figure
        plt.figure(figsize=(20, 20))
        
        # Calculate node sizes based on influence scores
        node_sizes = [self.influence_scores[node] * 10000 for node in self.G.nodes()]
        
        # Calculate edge weights for width
        edge_weights = [self.G[u][v]['weight'] for u, v in self.G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [0.5 + 3 * (w / max_weight) for w in edge_weights]
        
        # Create color map for communities
        num_communities = len(set(communities.values()))
        colors = plt.cm.rainbow(np.linspace(0, 1, num_communities))
        node_colors = [colors[communities[node]] for node in self.G.nodes()]
        
        # Compute layout with more space between nodes
        print("Computing layout...")
        layout = nx.spring_layout(self.G, 
                                k=2/np.sqrt(len(self.G.nodes())),  # Increased spacing
                                iterations=50,
                                seed=42)  # For reproducibility
        
        # Draw the network
        print("Drawing network...")
        
        # Draw edges
        nx.draw_networkx_edges(self.G, layout, 
                             alpha=0.2, 
                             width=edge_widths,
                             edge_color='gray')
        
        # Draw nodes
        nx.draw_networkx_nodes(self.G, layout,
                             node_size=node_sizes,
                             node_color=node_colors,
                             alpha=0.7)
        
        # Add labels for influential nodes
        # Only label nodes with high influence scores
        label_threshold = np.percentile(list(self.influence_scores.values()), 75)  # Top 25%
        labels = {node: node for node in self.G.nodes() 
                 if self.influence_scores[node] >= label_threshold}
        
        nx.draw_networkx_labels(self.G, layout, labels, 
                              font_size=8,
                              font_weight='bold')
        
        # Add title and legend
        plt.title("Twitter Interaction Network\n" +
                 f"Showing {len(self.G.nodes())} most influential users\n" +
                 "Node size: User influence | Colors: Communities | Edge width: Interaction strength",
                 pad=20)
        plt.axis('off')
        
        # Save the visualization
        print(f"Saving network visualization to {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_influence_report(self):
        """Generate a detailed report of influential users"""
        report = {
            'network_statistics': {
                'total_users': len(self.influence_scores),
                'influential_users': len(self.G.nodes()),
                'interactions': len(self.G.edges())
            },
            'top_influential_users': sorted(
                [{'user': user, 'influence_score': round(score, 4)}
                 for user, score in self.influence_scores.items()],
                key=lambda x: x['influence_score'],
                reverse=True
            )[:20]  # Top 20 users
        }
        
        # Save report
        with open('influence_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def main():
    # Initialize analyzer with higher influence threshold
    analyzer = TwitterNetworkAnalyzer('Election Tweets.json', min_influence_threshold=0.05)  # Adjust threshold as needed
    
    # Create and analyze network
    influence_scores = analyzer.create_filtered_network()
    
    # Detect communities in the filtered network
    communities = community_louvain.best_partition(analyzer.G)
    
    # Generate visualization
    analyzer.visualize_network(communities)
    
    # Generate influence report
    report = analyzer.generate_influence_report()
    
    print("\nAnalysis complete! Check the following files:")
    print("- network_graph.png (Network visualization)")
    print("- influence_analysis_report.json (Detailed metrics)")
    
    # Print top 10 influential users
    print("\nTop 10 Most Influential Users:")
    for user in report['top_influential_users'][:10]:
        print(f"- {user['user']}: {user['influence_score']}")

if __name__ == "__main__":
    main()
