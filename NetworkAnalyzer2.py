# install the following packages:
# pip install python-louvain networkx matplotlib numpy

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
    def __init__(self, file_path):
        """Initialize the analyzer with tweet data"""
        self.load_data(file_path)
        self.G = nx.Graph()
        
    def load_data(self, file_path):
        """Load tweet data from JSON file"""
        print("Loading tweet data...")
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.df = pd.DataFrame(data)
        print(f"Loaded {len(self.df)} tweets")

    def create_network(self):
        """Create network from tweet interactions"""
        print("Creating network from interactions...")
        
        # Track edges and weights
        edge_weights = {}
        
        for _, tweet in self.df.iterrows():
            try:
                # Get source user
                if isinstance(tweet.get('user'), dict):
                    source = tweet['user'].get('screen_name')
                    
                    # Add mentions
                    if isinstance(tweet.get('entities'), dict):
                        mentions = tweet['entities'].get('user_mentions', [])
                        for mention in mentions:
                            if isinstance(mention, dict):
                                target = mention.get('screen_name')
                                if source and target:
                                    edge = tuple(sorted([source, target]))
                                    edge_weights[edge] = edge_weights.get(edge, 0) + 1
                    
                    # Add reply relationships
                    if tweet.get('in_reply_to_screen_name'):
                        target = tweet['in_reply_to_screen_name']
                        if source and target:
                            edge = tuple(sorted([source, target]))
                            edge_weights[edge] = edge_weights.get(edge, 0) + 1
            
            except Exception as e:
                continue
        
        # Add edges to graph
        for (source, target), weight in edge_weights.items():
            self.G.add_edge(source, target, weight=weight)
        
        print(f"Created network with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")

    def analyze_network(self):
        """Perform network analysis"""
        print("Analyzing network...")
        
        metrics = {}
        
        # Basic metrics
        metrics['num_nodes'] = self.G.number_of_nodes()
        metrics['num_edges'] = self.G.number_of_edges()
        metrics['density'] = nx.density(self.G)
        
        # Degree metrics
        degrees = [d for n, d in self.G.degree()]
        metrics['avg_degree'] = np.mean(degrees)
        metrics['max_degree'] = max(degrees)
        
        # Centrality metrics
        degree_cent = nx.degree_centrality(self.G)
        betweenness_cent = nx.betweenness_centrality(self.G)
        
        # Find communities
        communities = community_louvain.best_partition(self.G)
        num_communities = len(set(communities.values()))
        metrics['num_communities'] = num_communities
        
        # Top users by different metrics
        metrics['top_users'] = {
            'by_degree': sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10],
            'by_betweenness': sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[:10]
        }
        
        return metrics, communities

    def visualize_network(self, communities, output_path='network_graph.png'):
        """Create and save network visualization"""
        print("Creating network visualization...")
        
        # Set up the figure
        plt.figure(figsize=(20, 20))
        
        # Calculate node sizes based on degree centrality
        degree_cent = nx.degree_centrality(self.G)
        node_sizes = [v * 5000 for v in degree_cent.values()]
        
        # Calculate edge weights for width
        edge_weights = [self.G[u][v]['weight'] for u, v in self.G.edges()]
        max_weight = max(edge_weights)
        edge_widths = [0.5 + 3 * (w / max_weight) for w in edge_weights]
        
        # Create color map for communities
        num_communities = len(set(communities.values()))
        colors = plt.cm.rainbow(np.linspace(0, 1, num_communities))
        node_colors = [colors[communities[node]] for node in self.G.nodes()]
        
        # Calculate layout
        print("Computing layout...")
        layout = nx.spring_layout(self.G, k=1/np.sqrt(len(self.G.nodes())), iterations=50)
        
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
        
        # Add labels for top nodes
        top_nodes = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:20]
        labels = {node: node for node, _ in top_nodes}
        nx.draw_networkx_labels(self.G, layout, labels, font_size=8)
        
        # Add title and adjust layout
        plt.title("Twitter Interaction Network\nNode size: User influence | Colors: Communities | Edge width: Interaction strength",
                 pad=20)
        plt.axis('off')
        
        # Save the visualization
        print(f"Saving network visualization to {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_network_report(self, metrics):
        """Generate a detailed report of network analysis"""
        report = {
            'network_statistics': {
                'nodes': metrics['num_nodes'],
                'edges': metrics['num_edges'],
                'density': round(metrics['density'], 4),
                'average_degree': round(metrics['avg_degree'], 2),
                'max_degree': metrics['max_degree'],
                'communities': metrics['num_communities']
            },
            'influential_users': {
                'by_connections': [
                    {'user': user, 'score': round(score, 4)} 
                    for user, score in metrics['top_users']['by_degree']
                ],
                'by_bridging': [
                    {'user': user, 'score': round(score, 4)} 
                    for user, score in metrics['top_users']['by_betweenness']
                ]
            }
        }
        
        # Save report
        with open('network_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def main():
    # Initialize analyzer
    analyzer = TwitterNetworkAnalyzer('Election Tweets.json')
    
    # Create and analyze network
    analyzer.create_network()
    metrics, communities = analyzer.analyze_network()
    
    # Generate visualization
    analyzer.visualize_network(communities)
    
    # Generate report
    report = analyzer.generate_network_report(metrics)
    
    print("\nAnalysis complete! Check the following files:")
    print("- network_graph.png (Network visualization)")
    print("- network_analysis_report.json (Detailed metrics)")

if __name__ == "__main__":
    main()
