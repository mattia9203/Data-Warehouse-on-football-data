import pandas as pd

main_data_path = "data/dataset_1/player_playing_time.csv"  # Replace with actual path
modified_player_valuations_path = "data/player_valuations_processed.csv"

# Load the datasets
main_data = pd.read_csv(main_data_path)
modified_player_valuations = pd.read_csv(modified_player_valuations_path)

# Extract the player_name columns from both datasets
main_players = main_data['player'].dropna().unique()
modified_players = modified_player_valuations['player_name'].dropna().unique()

# Find common players between the two datasets
common_players = set(main_players) & set(modified_players)


num_main_players = len(main_players)
num_modified_players = len(modified_players)

# Count of common players
num_common_players = len(common_players)

# Output the results
print(f"Number of players in the main dataset: {num_main_players}")
print(f"Number of players in the modified player valuations dataset: {num_modified_players}")
print(f"Number of common players between the main dataset and modified player valuations dataset: {num_common_players}")
print("\nList of common players:")