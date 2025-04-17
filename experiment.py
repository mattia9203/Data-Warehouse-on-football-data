import pandas as pd

player_valuations_path = "data/updated_player_valuations_fuzzy.csv"
dataset_1_path = "data/dataset_1/player_possession.csv"

player_valuations = pd.read_csv(player_valuations_path)
dataset_1 = pd.read_csv(dataset_1_path)

# Get the unique club names from both datasets
club_names_dataset_2 = player_valuations['club_name'].dropna().unique()
club_names_dataset_1 = dataset_1['squad'].dropna().unique()

# Find common clubs between the two datasets
common_clubs = set(club_names_dataset_2) & set(club_names_dataset_1)

# Compute the percentage of common clubs in dataset_2
common_percentage = (len(common_clubs) / len(club_names_dataset_2)) * 100 if len(club_names_dataset_2) > 0 else 0

# Identify clubs in dataset_2 but not in dataset_1 and vice versa
clubs_not_in_dataset_1 = set(club_names_dataset_2) - set(club_names_dataset_1)
clubs_not_in_dataset_2 = set(club_names_dataset_1) - set(club_names_dataset_2)

# Print results
print(f"Percentage of common clubs between both datasets: {common_percentage:.2f}%")
print("\nClubs in dataset_2 but not in dataset_1:")
print(clubs_not_in_dataset_1)

print("\nClubs in dataset_1 but not in dataset_2:")
print(clubs_not_in_dataset_2)

# Load both datasets
dataset_1_path = "data/dataset_1/player_possession.csv"  # Adjust path
dataset_2_path = "data/updated_player_valuations_with_year.csv"  # Adjust path

dataset_1 = pd.read_csv(dataset_1_path)
dataset_2 = pd.read_csv(dataset_2_path)

# Extract relevant columns: 'player' and 'season' for dataset_1, 'player_name' and 'valuation_year' for dataset_2
dataset_1_pairs = dataset_1[['player', 'season']].dropna()
dataset_2_pairs = dataset_2[['player_name', 'year']].dropna()

# Find common pairs between both datasets
common_pairs = pd.merge(dataset_1_pairs, dataset_2_pairs, left_on=['player', 'season'], right_on=['player_name', 'year'], how='inner')

# Count the number of common pairs
num_common_pairs = common_pairs.shape[0]

# Output the result
print(f"Number of common year-player name pairs in both datasets: {num_common_pairs}")

# Optionally, show the first few common pairs for verification
print("\nExample of common year-player pairs:")
print(common_pairs.head())