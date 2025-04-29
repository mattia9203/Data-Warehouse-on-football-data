import pandas as pd
from rapidfuzz import process
import os
import re

# Load the files
players = pd.read_csv("data/dataset_2/players.csv")
clubs = pd.read_csv("data/dataset_2/clubs.csv")
competitions = pd.read_csv("data/dataset_2/competitions.csv")
player_valuations = pd.read_csv("data/dataset_2/player_valuations.csv")
"""
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
valu.to_csv(os.path.join(DATA_DIR, "valuations_with_season_club.csv"), index=False)"""

# ─── paths ──────────────────────────────────────────────────────────
DATA1_DIR = "data/dataset_1"
DATA2_DIR = "data/dataset_2"
dataset1_files = [
    "player_defense.csv", "player_gca.csv", "player_misc.csv",
    "player_shooting.csv", "player_possession.csv",
    "player_passing_type.csv", "player_passing.csv",
    "player_standard_stats.csv"
]

players_path = os.path.join(DATA2_DIR, "players.csv")

# ─── load players reference ─────────────────────────────────────────
players_df = pd.read_csv(players_path)

# Combine first and last names into player_name
players_df['player_name'] = (
    players_df['first_name'].fillna('') + ' ' + players_df['last_name'].fillna('')
).str.strip()

# Parse full datetime and extract just the year
players_df['date_of_birth'] = pd.to_datetime(
    players_df['date_of_birth'], errors='coerce'
)
players_df['year_of_birth'] = players_df['date_of_birth'].dt.year

# 2) Build lookup table indexed by player_name
players_ref = players_df.set_index('player_name')[[
    'country_of_citizenship',   # for nation
    'country_of_birth',          # for country
    'year_of_birth'              # for born
]]

# 3) Process each stats file
for fname in dataset1_files:
    in_path = os.path.join(DATA1_DIR, fname)
    if not os.path.exists(in_path):
        print(f"⚠️  {fname} not found under {DATA1_DIR}, skipping.")
        continue

    df = pd.read_csv(in_path)

    # Merge the reference on player_name
    df = df.merge(
        players_ref,
        how='left',
        left_on='player',
        right_index=True
    )

    # Fill missing nation → citizenship → birth country
    df['nation'] = (
        df['nation']
          .fillna(df['country_of_citizenship'])
          .fillna(df['country_of_birth'])
    )

    # Fill missing country → birth country
    df['country'] = df['country'].fillna(df['country_of_birth'])

    # Fill missing born → birth year
    df['born'] = df['born'].fillna(df['year_of_birth'])

    # Only drop helper columns that actually exist
    helper_cols = [
        'country_of_citizenship',
        'country_of_birth',
        'date_of_birth',
        'year_of_birth'
    ]
    cols_to_drop = [c for c in helper_cols if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Save cleaned output
    out_path = os.path.join(DATA1_DIR, f"cleaned_{fname}")
    df.to_csv(out_path, index=False)
    print(f"✅ Processed {fname} → saved cleaned_{fname}")

dataset1_files = [
    "cleaned_player_defense.csv", "cleaned_player_gca.csv", "cleaned_player_misc.csv",
    "cleaned_player_shooting.csv", "cleaned_player_possession.csv",
    "cleaned_player_passing_type.csv", "cleaned_player_passing.csv",
    "cleaned_player_standard_stats.csv"
]

for fname in dataset1_files:
    path = os.path.join(DATA1_DIR, fname)
    if not os.path.exists(path):
        print(f"⚠️  {fname} not found, skipping.")
        continue

    df = pd.read_csv(path)

    # Ensure born and season are numeric
    df['born']   = pd.to_numeric(df['born'],   errors='coerce')
    df['season'] = pd.to_numeric(df['season'], errors='coerce')

    # mask: rows with missing age but valid born & season
    mask = df['age'].isna() & df['born'].notna() & df['season'].notna()

    # compute age
    df.loc[mask, 'age'] = df.loc[mask, 'season'] - df.loc[mask, 'born']
    
    # save back
    df.to_csv(path, index=False)
    print(f"✅ {fname}: filled {mask.sum()} age values")
    
    # Phase A: build a country → continent dictionary from existing data
# ═══════════════════════════════════════════════════════════════════════════
country_to_continent = {}
for fname in dataset1_files:
    path = os.path.join(DATA1_DIR, fname)
    if not os.path.exists(path):
        continue
    tmp = pd.read_csv(path, usecols=['country', 'continent']).dropna()
    country_to_continent.update(
        pd.Series(tmp.continent.values, index=tmp.country).to_dict()
    )
print(f"Lookup built: {len(country_to_continent)} country→continent pairs")

# ═════════════════════════════════════════════════════════════════════
# Phase B – fill country & continent; drop rows missing born
# ═════════════════════════════════════════════════════════════════════
players = pd.read_csv(players_path)
players['player_name'] = (players['first_name'].fillna('') + ' ' +
                          players['last_name'].fillna('')).str.strip()
lookup_players = players.set_index('player_name')[['country_of_birth']]

for fname in dataset1_files:
    in_path = os.path.join(DATA1_DIR, fname)
    if not os.path.exists(in_path):
        print(f"⚠️  {fname} not found, skipping")
        continue

    df = pd.read_csv(in_path)

    # Merge country_of_birth
    df = df.merge(lookup_players, how='left',
                  left_on='player', right_index=True)

    # -------- fill COUNTRY -------------------------------------------
    before_country_na = df['country'].isna().sum()
    df['country'] = df['country'].fillna(df['country_of_birth'])
    country_filled = before_country_na - df['country'].isna().sum()

    # -------- fill CONTINENT -----------------------------------------
    mask_continent = df['continent'].isna() & df['country'].notna()
    before_continent_na = df['continent'].isna().sum()
    df.loc[mask_continent, 'continent'] = df.loc[mask_continent, 'country'] \
        .map(country_to_continent)
    continent_filled = before_continent_na - df['continent'].isna().sum()

    # Drop helper column
    df.drop(columns=['country_of_birth'], inplace=True)

    # -------- drop rows still missing born ---------------------------
    before_rows = len(df)
    df = df.dropna(subset=['born'])
    dropped_rows = before_rows - len(df)

    # Save cleaned file
    out_path = os.path.join(DATA1_DIR, f"{fname}")
    df.to_csv(out_path, index=False)

    # Report
    print(f"✅ {fname}: +{country_filled} country, +{continent_filled} continent "
          f"filled; dropped {dropped_rows} rows  → saved cleaned2_{fname}")


OUTPUT_FILE = "data/top8000_final_sample.csv"
DATA2_FILE = "data/valuations_with_season_club.csv"
OUTPUT_DIR1 = "data/dataset_1/selected_8000_stats"
OUTPUT_DIR2 = "data/selected_8000"
TOP_N = 8000
DATA_DIR = "data"

# --------------------------------------------------------------------

MASTER_FILE = "cleaned_player_standard_stats.csv"   # used to rank nulls
# --------------------------------------------------------------------

os.makedirs(OUTPUT_DIR1, exist_ok=True)
os.makedirs(OUTPUT_DIR2, exist_ok=True)

# 1 ── load the FULL valuation file (all columns) and build the pair-set
val_df = pd.read_csv(DATA2_FILE)                # <-- no usecols
val_pairs = set(zip(val_df['player_name'], val_df['year']))
print(f"Pairs in valuations: {len(val_pairs):,}")

# 2 ── Load MASTER stats file to score nulls
master_path = os.path.join(DATA1_DIR, MASTER_FILE)
master_df = pd.read_csv(master_path)

# keep rows with country & continent and that exist in valuations
mask = (
    master_df['country'].notna() &
    master_df['continent'].notna() &
    [(p, s) in val_pairs for p, s in zip(master_df['player'],
                                        master_df['season'])]
)
master_df = master_df.loc[mask].copy()

# compute per-row missing count (excluding key columns)
feature_cols = [c for c in master_df.columns if c not in ('player', 'season')]
master_df['missing'] = master_df[feature_cols].isna().sum(axis=1)

# rank and take best TOP_N pairs
top_pairs = (
    master_df
      .sort_values('missing')
      .head(TOP_N)[['player', 'season']]
)
pair_set = set(zip(top_pairs['player'], top_pairs['season']))
print(f"Selected {len(pair_set)} best pairs")

# 3 ── Filter every stats file to those pairs
for fname in dataset1_files:
    src = os.path.join(DATA1_DIR, fname)
    if not os.path.exists(src):
        print(f"⚠️  {fname} missing, skipped")
        continue
    df = pd.read_csv(src)
    df_sel = df.loc[
        [(p, s) in pair_set for p, s in zip(df['player'], df['season'])]
    ]
    out = os.path.join(OUTPUT_DIR1, f"selected_{fname}")
    df_sel.to_csv(out, index=False)
    print(f"  ↳ {fname}: kept {len(df_sel):,} rows → {out}")

# 4 ── filter the FULL valuation DataFrame and keep every column
val_sel = val_df.loc[
    [(p, y) in pair_set for p, y in zip(val_df['player_name'],
                                       val_df['year'])]
]
val_out = os.path.join(OUTPUT_DIR2, "selected_valuations.csv")
val_sel.to_csv(val_out, index=False)
print(f"Valuations kept {len(val_sel):,} full rows → {val_out}")