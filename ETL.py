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

# Create a full name column, handle missing first names
player_valuations['player_name'] = player_valuations['first_name'].fillna('') + " " + player_valuations['last_name']

# If the first name is missing and last name is present, we ensure there's no leading space
player_valuations['player_name'] = player_valuations['player_name'].str.strip()

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

manual_mapping_competition = {
    'premier-league': 'Premier League',
    'serie-a': 'Serie A',
    'laliga': 'La Liga',
    'bundesliga': 'Bundesliga',
    'ligue-1': 'Ligue 1',
}

# Function to replace competition names based on manual mapping
def replace_competition_name(competition_name):
    # Check if the competition is in the manual mapping
    if competition_name in manual_mapping_competition:
        return manual_mapping_competition[competition_name]
    return None  # If not in the manual mapping, return None to drop it

# Apply the mapping to the 'competition_name' column in player_valuations
player_valuations['competition_name'] = player_valuations['competition_name'].apply(replace_competition_name)

# Drop rows where the competition name is not in the manual mapping (i.e., None)
player_valuations = player_valuations.dropna(subset=['competition_name'])

# Save the cleaned dataset with the correct competition names
cleaned_player_valuations_path = "data/player_valuations_with_competitions.csv"
#player_valuations.to_csv(cleaned_player_valuations_path, index=False)

player_valuations['date'] = pd.to_datetime(player_valuations['date'], errors='coerce')

# Remove rows where the valuation year is before 2018
player_valuations = player_valuations[player_valuations['date'].dt.year >= 2018]

# Save the cleaned dataset (with only valuations from 2018 and onward)
cleaned_player_valuations_path = "data/player_valuations_2018.csv"
#player_valuations.to_csv(cleaned_player_valuations_path, index=False)


# Extract the unique club names from dataset_1
dataset_1_clubs = dataset_1['squad'].unique()
dataset_1_competitions = dataset_1['comp'].unique()

# List of common terms to remove
common_terms = ['Club', 'FC', 'SC', 'Associazione', 'Sportiva', 'De', 'Royal', ]
manual_mapping_clubs = {
    'Stade Rennais Football' : 'Rennes',
    'Manchester United Football' : 'Manchester Utd',
    'Stade brestois 29' : 'Brest',
    'Manchester City Football' : 'Manchester City',
    "Olympique Gymnaste Nice Côte d'Azur" :  'Nice',
    'Wolverhampton Wanderers Football' : 'Wolves',
    'Athletic Bilbao' : 'Athletic Club',
    'Reial Deportiu Espanyol Barcelona S.A.D.' : 'Espanyol',
    'Verein für Leibesübungen Bochum 1848 Fußballgemeinschaft' : 'Bochum',
    'Newcastle United Football' : 'Newcastle Utd',
    '1. Nuremberg' : 'Nürnberg',
    'Bayern München' : 'Bayern Munich',
    'Borussia Verein für Leibesübungen 1900 Mönchengladbach' : 'Gladbach',
    'Le Havre Athletic' : 'Le Havre',
    'Association la Jeunesse auxerroise' : 'Auxerre'
}
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

def manual_mapping_check(club_name):
    # Check if the club is in the manual mapping dictionary
    if club_name in manual_mapping_clubs:
        return manual_mapping_clubs[club_name]
    return None  # If not in the manual mapping, return None to proceed to fuzzy matching

# Function to perform fuzzy matching on club names
def fuzzy_match_club(name, choices):
    match = process.extractOne(name, choices)
    if match and match[1] > 85:  # Match score threshold (adjust if needed)
        return match[0]
    return name  # If no good match, return original name

# Function to handle the full matching process
def match_club_name(club_name, choices):
    # First check if the club name is in the manual mapping
    mapped_name = manual_mapping_check(club_name)
    
    if mapped_name:
        return mapped_name  # If found in the manual mapping, return the mapped name
    
    # If not in manual mapping, proceed with fuzzy matching
    return fuzzy_match_club(club_name, choices)

# Apply fuzzy matching to replace club names in player_valuations
player_valuations['club_name'] = player_valuations['club_name_cleaned'].apply(
    lambda x: match_club_name(x, dataset_1_clubs_cleaned)
)

# Drop the temporary cleaned column
player_valuations.drop(columns=['club_name_cleaned'], inplace=True)

# Save the updated dataset with correct club names
updated_player_valuations_path = "data/updated_player_valuations_fuzzy.csv"
#player_valuations.to_csv(updated_player_valuations_path, index=False)

player_valuations_path = "data/updated_player_valuations_fuzzy.csv"
player_valuations = pd.read_csv(player_valuations_path)

# Convert the date column to datetime format (replace 'date_column' with your actual column name)
player_valuations['date'] = pd.to_datetime(player_valuations['date'], errors='coerce')

# Extract the year from the datetime column
player_valuations['year'] = player_valuations['date'].dt.year

# Drop the original 'valuation_date' column if no longer needed
player_valuations.drop(columns=['date'], inplace=True)

# Save the updated dataset
updated_player_valuations_path = "data/updated_player_valuations_with_year.csv"
#player_valuations.to_csv(updated_player_valuations_path, index=False)
# ---------- paths ----------
DATA_DIR   = "data"
VAL_FILE   = os.path.join(DATA_DIR, "updated_player_valuations_with_year.csv")
DATA_DIR   = "data/dataset_1"
STAT_FILES = [  # all season‑stat csvs               ↓ add/remove as needed
    "player_defense.csv", "player_gca.csv",
    "player_misc.csv",    "player_shooting.csv",
    "player_possession.csv", "player_passing_type.csv",
    "player_passing.csv",    "player_standard_stats.csv"
]
STAT_FILES = [os.path.join(DATA_DIR, f) for f in STAT_FILES]
# --------------------------------------------------------------------------

# 1) build a ***single lookup table***  (player, year)  -> squad
pairs = []                       # collect mini‑tables, then concat once
for path in STAT_FILES:
    df = pd.read_csv(path, usecols=["player", "season", "squad"])
    df.rename(columns={"player":"player_name",
                       "season":"year",
                       "squad":"club_in_year"}, inplace=True)
    pairs.append(df.drop_duplicates())

lookup = pd.concat(pairs, ignore_index=True).drop_duplicates()

# 2) load valuations and merge the club of that season
valu = pd.read_csv(VAL_FILE)
valu = valu.merge(lookup, on=["player_name", "year"], how="left")

# 3) fill gaps with CURRENT club_name if seasonal club missing
valu["club_in_year"] = valu["club_in_year"].fillna(valu["club_name"])

# 4) report & save
filled = valu["club_in_year"].notna().sum()
print(f"club_in_year filled for {filled} of {len(valu)} rows (including fall‑back to current club)")
valu.to_csv(os.path.join(DATA_DIR, "valuations_with_season_club.csv"), index=False)