import pandas as pd
from rapidfuzz import process
import os
import re

# Load the files
players = pd.read_csv("data/dataset_2/players.csv")
clubs = pd.read_csv("data/dataset_2/clubs.csv")
competitions = pd.read_csv("data/dataset_2/competitions.csv")
player_valuations = pd.read_csv("data/dataset_2/player_valuations.csv")

# Merge the player_valuations dataset with player names, clubs, and competitions
# Merge player names (player_id -> full_name)
player_valuations = pd.merge(player_valuations, players[['player_id', 'first_name', 'last_name']], on='player_id', how='left')

# Create a full name column
player_valuations['player_name'] = player_valuations['first_name'] + " " + player_valuations['last_name']

# Drop the original first_name and last_name columns
player_valuations.drop(columns=['first_name', 'last_name'], inplace=True)

# Merge club names (current_club_id -> club_name)
player_valuations = pd.merge(player_valuations, clubs[['club_id', 'name']], left_on='current_club_id', right_on='club_id', how='left')

# Rename club name column
player_valuations.rename(columns={'name': 'club_name'}, inplace=True)

# Merge competition names (player_club_domestic_competition_id -> competition_name)
player_valuations = pd.merge(player_valuations, competitions[['competition_id', 'name']], left_on='player_club_domestic_competition_id', right_on='competition_id', how='left')

# Rename competition column
player_valuations.rename(columns={'name': 'competition_name'}, inplace=True)

# Drop the original ID columns after merging
player_valuations.drop(columns=['player_id', 'current_club_id', 'player_club_domestic_competition_id', 'club_id', 'competition_id'], inplace=True)

# Save the processed file
player_valuations.to_csv("data/player_valuations_processed.csv", index=False)

player_valuations_path = "data/player_valuations_processed.csv"
player_valuations = pd.read_csv(player_valuations_path)
dataset_1_path = "data/dataset_1/player_possession.csv"
dataset_1 = pd.read_csv(dataset_1_path)


# Extract the unique club names from dataset_1
dataset_1_clubs = dataset_1['squad'].unique()
dataset_1_competitions = dataset_1['comp'].unique()

# List of common terms to remove
common_terms = ['Club', 'FC', 'SC', 'Associazione', 'Sportiva', 'Real', 'De', 'Royal', 'United']

# Function to clean up club names by removing common terms
def clean_club_name(name):
    # Remove common terms (case-insensitive)
    name = re.sub(r'\b(?:' + '|'.join(common_terms) + r')\b', '', name, flags=re.IGNORECASE)
    # Remove any extra spaces that may appear after term removal
    name = ' '.join(name.split())  # This removes extra spaces
    return name

# Clean club names in both datasets
dataset_1_clubs_cleaned = [clean_club_name(club) for club in dataset_1_clubs]
player_valuations['club_name_cleaned'] = player_valuations['club_name'].apply(lambda x: clean_club_name(x))

# Function to perform fuzzy matching on club names
def fuzzy_match_club(name, choices):
    match = process.extractOne(name, choices)
    if match and match[1] > 80:  # Match score threshold (adjust if needed)
        return match[0]
    return name  # If no good match, return original name

# Apply fuzzy matching to replace club names in player_valuations
#player_valuations['club_name'] = player_valuations['club_name_cleaned'].apply(
#    lambda x: fuzzy_match_club(x, dataset_1_clubs_cleaned)
#)

# Drop the temporary cleaned column
#player_valuations.drop(columns=['club_name_cleaned'], inplace=True)

# Save the updated dataset with correct club names
#updated_player_valuations_path = "data/updated_player_valuations_fuzzy.csv"
#player_valuations.to_csv(updated_player_valuations_path, index=False)

updated_player_valuations_path = "data/updated_player_valuations_fuzzy.csv"
player_valuations = pd.read_csv (updated_player_valuations_path)
# Convert the valuation_date column to datetime (if it's not already)
player_valuations['date'] = pd.to_datetime(player_valuations['date'], errors='coerce')

# Remove rows where the valuation year is before 2018
player_valuations = player_valuations[player_valuations['date'].dt.year >= 2018]

# Save the cleaned dataset (with only valuations from 2018 and onward)
cleaned_player_valuations_path = "data/cleaned_player_valuations_2018.csv"
player_valuations.to_csv(cleaned_player_valuations_path, index=False)
