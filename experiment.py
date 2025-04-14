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

updated_player_valuations_path = "data/cleaned_player_valuations_2018.csv"
updated_player_valuations = pd.read_csv(updated_player_valuations_path)

player_valuations_path = "data/player_valuations_processed.csv"
player_valuations = pd.read_csv(player_valuations_path)
player_valuations['date'] = pd.to_datetime(player_valuations['date'], errors='coerce')

# Remove rows where the valuation year is before 2018
player_valuations = player_valuations[player_valuations['date'].dt.year >= 2018]

# Save the cleaned dataset (with only valuations from 2018 and onward)
player_valuations_path = "data/player_valuations_2018.csv"
player_valuations.to_csv(player_valuations_path, index=False)
# Preview the first few rows of the updated dataset
print(updated_player_valuations.head(20))
print(player_valuations.head(20))