
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import os  # Added missing import

class UserInfluenceAnalyzer:
    def __init__(self, file_path):
        """Initialize analyzer with tweet data"""
        # Create output directory
        self.output_dir = 'influence_analysis'
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_data(file_path)
    
    def load_data(self, file_path):
        """Load tweet data"""
        try:
            print("Loading tweet data...")
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            self.df = pd.DataFrame(data)
            print(f"Loaded {len(self.df)} tweets")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def analyze_user_influence(self):
        """Analyze user influence patterns"""
        print("Analyzing user influence...")
        
        # Extract user metrics
        user_metrics = []
        for _, tweet in self.df.iterrows():
            try:
                if isinstance(tweet.get('user'), dict):
                    user = tweet['user']
                    metrics = {
                        'screen_name': user.get('screen_name', 'unknown'),
                        'followers_count': int(user.get('followers_count', 0)),
                        'friends_count': int(user.get('friends_count', 0)),
                        'statuses_count': int(user.get('statuses_count', 0)),
                        'engagement': int((tweet.get('favorite_count', 0) or 0) + 
                                    (tweet.get('retweet_count', 0) or 0)),
                        'verified': bool(user.get('verified', False))
                    }
                    user_metrics.append(metrics)
            except Exception as e:
                print(f"Error processing tweet: {str(e)}")
                continue
        
        # Convert to DataFrame
        user_df = pd.DataFrame(user_metrics)
        
        # Check if DataFrame is empty
        if user_df.empty:
            print("No valid user metrics found!")
            return pd.DataFrame()
        
        # Remove duplicates (keep latest entry for each user)
        user_df = user_df.drop_duplicates(subset='screen_name', keep='last')
        
        # Add small constant to avoid log(0)
        user_df['followers_count'] = user_df['followers_count'] + 1
        user_df['engagement'] = user_df['engagement'] + 1
        user_df['statuses_count'] = user_df['statuses_count'] + 1
        
        # Calculate influence score
        user_df['influence_score'] = (
            0.4 * np.log1p(user_df['followers_count']) +
            0.3 * np.log1p(user_df['engagement']) +
            0.3 * np.log1p(user_df['statuses_count'])
        )
        
        # Categorize users
        def categorize_user(row):
            if row['verified']:
                return 'Verified'
            elif row['followers_count'] > 10000:
                return 'High Impact'
            elif row['followers_count'] > 1000:
                return 'Medium Impact'
            else:
                return 'Regular User'
        
        user_df['category'] = user_df.apply(categorize_user, axis=1)
        
        # Create visualizations
        self.create_influence_visualizations(user_df)
        
        return user_df
    
    def filter_users_for_visualization(self, user_df, min_followers=100, min_engagement=5, top_percentage=20):
        """
        Filter users for visualization based on multiple criteria
        
        Parameters:
        - min_followers: minimum number of followers to include
        - min_engagement: minimum engagement score to include
        - top_percentage: top percentage of users by influence score to include
        """
        # First filter by minimum criteria
        filtered_df = user_df[
            (user_df['followers_count'] >= min_followers) & 
            (user_df['engagement'] >= min_engagement)
        ]
        
        # Then take top percentage by influence score
        threshold = np.percentile(filtered_df['influence_score'], 100 - top_percentage)
        final_df = filtered_df[filtered_df['influence_score'] >= threshold]
        
        return final_df

    # Modify the scatter plot part in create_influence_visualizations
    def create_influence_visualizations(self, user_df):
        """Create visualizations for user influence analysis"""
        print("Creating visualizations...")
        try:
            # Set the style parameters
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
            plt.rcParams['axes.labelsize'] = 12
            plt.rcParams['axes.titlesize'] = 14
            plt.rcParams['xtick.labelsize'] = 10
            plt.rcParams['ytick.labelsize'] = 10
            
            # 1. Scatter plot with filtered data
            plt.figure(figsize=(12, 8))
            
            # Filter users for visualization
            filtered_df = self.filter_users_for_visualization(
                user_df,
                min_followers=100,    # Minimum followers to include
                min_engagement=5,     # Minimum engagement to include
                top_percentage=20     # Top 20% of influential users
            )
            
            print(f"Plotting {len(filtered_df)} users out of {len(user_df)} total users")
            
            # Create main scatter plot
            scatter = plt.scatter(
                filtered_df['followers_count'],
                filtered_df['engagement'],
                c=filtered_df['influence_score'],
                s=filtered_df['influence_score'] * 20,  # Size based on influence
                alpha=0.6,
                cmap='viridis'
            )
            
            # Add colorbar with custom label
            cbar = plt.colorbar(scatter)
            cbar.set_label('Influence Score', rotation=270, labelpad=15)
            
            # Set scales and labels
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Followers Count (log scale)')
            plt.ylabel('Engagement (log scale)')
            plt.title(f'User Influence: Top {len(filtered_df)} Most Influential Users', pad=20)
            plt.grid(True, alpha=0.3)
            
            # Add annotations for top users only
            top_users = filtered_df.nlargest(10, 'influence_score')
            for _, user in top_users.iterrows():
                plt.annotate(
                    user['screen_name'],
                    (user['followers_count'], user['engagement']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(
                        facecolor='white',
                        edgecolor='gray',
                        alpha=0.8,
                        pad=0.5
                    )
                )
            
            # Add legend for user categories
            category_colors = {
                'Verified': 'red',
                'High Impact': 'orange',
                'Medium Impact': 'green',
                'Regular User': 'blue'
            }
            
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', 
                        markerfacecolor=color, label=cat, markersize=8)
                for cat, color in category_colors.items()
                if cat in filtered_df['category'].unique()
            ]
            plt.legend(handles=legend_elements, 
                    title='User Categories',
                    loc='upper left',
                    bbox_to_anchor=(1.15, 1))
            
            # Add text box with statistics
            stats_text = (
                f"Total Users: {len(user_df):,}\n"
                f"Filtered Users: {len(filtered_df):,}\n"
                f"Avg Influence: {filtered_df['influence_score'].mean():.2f}\n"
                f"Max Influence: {filtered_df['influence_score'].max():.2f}"
            )
            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
                    verticalalignment='top',
                    fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/influence_scatter.png', 
                    dpi=300, 
                    bbox_inches='tight',
                    pad_inches=0.2)
            plt.close()
            
            # Add filtering information to stats
            stats = {
                'visualization_filters': {
                    'total_users': len(user_df),
                    'filtered_users': len(filtered_df),
                    'min_followers': 100,
                    'min_engagement': 5,
                    'top_percentage': 20
                },
                'filtered_user_stats': {
                    'avg_influence': filtered_df['influence_score'].mean().round(2),
                    'avg_followers': filtered_df['followers_count'].mean().round(2),
                    'avg_engagement': filtered_df['engagement'].mean().round(2)
                }
            }
            
            with open(f'{self.output_dir}/scatter_plot_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
                
            print(f"Created scatter plot with {len(filtered_df)} filtered users")
            
            # Continue with other visualizations...
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    try:
        analyzer = UserInfluenceAnalyzer('Election Tweets.json')
        user_df = analyzer.analyze_user_influence()
        
        if not user_df.empty:
            print("\nAnalysis complete! Check the following files in the 'influence_analysis' directory:")
            print("- influence_scatter.png (Followers vs Engagement visualization)")
            print("- engagement_by_category.png (Engagement distribution by user category)")
            print("- avg_influence_by_category.png (Average influence scores)")
            print("- influence_stats.json (Detailed statistics)")
            
            # Print summary of top influential users
            print("\nTop 10 Most Influential Users:")
            top_users = user_df.nlargest(10, 'influence_score')
            for _, user in top_users.iterrows():
                print(f"- {user['screen_name']}: {user['category']} "
                      f"(Influence Score: {user['influence_score']:.2f})")
        else:
            print("Analysis failed: No valid user data found")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
