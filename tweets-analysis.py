import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
import re
import os

# Set style for better visualizations
sns.set_theme()

def load_tweets(file_path):
    """
    Load tweets from JSON file with proper JSON format handling
    """
    try:
        print(f"Attempting to open file: {file_path}")
        
        # Read the entire file content first
        with open(file_path, 'r', encoding='utf-8') as file:
            print("File opened successfully, reading contents...")
            # Load the entire JSON content
            data = json.load(file)
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
                print(f"Successfully loaded {len(df)} tweets")
                return df
            else:
                print("Error: JSON file is not in the expected list format")
                return pd.DataFrame()
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print("Available files in current directory:")
        for file in os.listdir():
            print(f"- {file}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error loading file: {str(e)}")
        return pd.DataFrame()

# Main analysis
if __name__ == "__main__":
    # Try different possible file names
    possible_files = ['Election Tweets.json', 'election tweets.json', 'Election_Tweets.json', 'tweets.json']
    
    df = pd.DataFrame()
    for file_name in possible_files:
        print(f"\nTrying to load: {file_name}")
        if os.path.exists(file_name):
            df = load_tweets(file_name)
            if not df.empty:
                print(f"Successfully loaded data from {file_name}")
                break
    
    if df.empty:
        print("\nCould not load data from any of the expected file names.")
        print("Please enter the exact file name (including extension): ")
        custom_file = input()
        if os.path.exists(custom_file):
            df = load_tweets(custom_file)
    
    if df.empty:
        print("No data loaded. Exiting program.")
        exit()

    # Display initial data info
    print("\nDataset Information:")
    print(df.info())
    print("\nSample columns:", df.columns.tolist())

    # Basic statistics
    print("\nCalculating basic statistics...")
    try:
        basic_stats = {
            'Total Tweets': len(df),
            'Unique Users': df['user'].apply(lambda x: x['screen_name'] if isinstance(x, dict) else x).nunique(),
            'Date Range': {
                'Start': min(pd.to_datetime(df['created_at'])),
                'End': max(pd.to_datetime(df['created_at']))
            }
        }
        print("\nBasic Statistics:")
        for key, value in basic_stats.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error calculating basic stats: {str(e)}")

    # User classification with error handling
    def classify_user(user):
        try:
            if not isinstance(user, dict):
                return 'Unknown'
            
            if user.get('verified'):
                desc = user.get('description', '').lower()
                if 'journalist' in desc or 'reporter' in desc:
                    return 'Journalist'
                elif any(term in desc for term in ['mp', 'minister', 'party']):
                    return 'Political Figure'
                return 'Verified Other'
            elif user.get('followers_count', 0) > 10000:
                return 'Influencer'
            elif user.get('statuses_count', 0) > 50000:
                return 'High Volume Account'
            return 'Citizen'
        except:
            return 'Unknown'

    print("\nClassifying users...")
    df['actor_type'] = df['user'].apply(classify_user)

    # Create output directory
    os.makedirs('analysis_output', exist_ok=True)

    # Plot user distribution
    try:
        plt.figure(figsize=(10, 6))
        df['actor_type'].value_counts().plot(kind='bar')
        plt.title('Distribution of User Types')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('analysis_output/user_distribution.png')
        plt.close()
    except Exception as e:
        print(f"Error creating user distribution plot: {str(e)}")

    # Migration analysis
    print("\nAnalyzing migration discourse...")
    migration_terms = ['migration', 'immigrant', 'refugee', 'asylum', 'border',
                      'newcomer', 'immigration', 'migrant']
    
    df['is_migration'] = df['full_text'].apply(
        lambda x: any(term in str(x).lower() for term in migration_terms)
    )

    # Hashtag analysis with error handling
    def extract_hashtags(entities):
        try:
            if isinstance(entities, dict) and 'hashtags' in entities:
                return [tag['text'].lower() for tag in entities['hashtags']]
        except:
            pass
        return []

    try:
        migration_tweets = df[df['is_migration']]
        hashtags = [tag for tags in migration_tweets['entities'].apply(extract_hashtags) 
                   for tag in tags]
        top_hashtags = pd.Series(hashtags).value_counts().head(10)

        # Plot hashtags
        plt.figure(figsize=(10, 6))
        top_hashtags.plot(kind='bar')
        plt.title('Top Hashtags in Migration-Related Tweets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('analysis_output/top_hashtags.png')
        plt.close()
    except Exception as e:
        print(f"Error in hashtag analysis: {str(e)}")

    # Save results
    print("\nSaving analysis results...")
    try:
        analysis_results = {
            'basic_stats': basic_stats,
            'actor_distribution': df['actor_type'].value_counts().to_dict(),
            'migration_stats': {
                'total_migration_tweets': int(df['is_migration'].sum()),
                'percentage_migration': float((df['is_migration'].mean() * 100).round(2)),
                'top_hashtags': top_hashtags.to_dict() if 'top_hashtags' in locals() else {}
            }
        }

        with open('analysis_output/analysis_results.json', 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)

        print("\nAnalysis complete! Results saved in 'analysis_output' directory:")
        print("- user_distribution.png")
        print("- top_hashtags.png")
        print("- analysis_results.json")
    except Exception as e:
        print(f"Error saving results: {str(e)}")