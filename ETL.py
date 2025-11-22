import pandas as pd
#from rapidfuzz import process
import os
import re
from sqlalchemy import create_engine

"""
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
    # Remove common terms
    name = re.sub(r'\b(?:' + '|'.join(common_terms) + r')\b', '', name, flags=re.IGNORECASE)
    # Remove any extra spaces that may appear after term removal
    name = ' '.join(name.split()) 
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
    if match and match[1] > 85:  # Match score threshold 
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

# 1) build a single lookup table (player, year)  -> squad
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

filled = valu["club_in_year"].notna().sum()
print(f"club_in_year filled for {filled} of {len(valu)}")
valu.to_csv(os.path.join(DATA_DIR, "valuations_with_season_club.csv"), index=False)

DATA1_DIR = "data/dataset_1"
DATA2_DIR = "data/dataset_2"
dataset1_files = [
    "player_defense.csv", "player_gca.csv", "player_misc.csv",
    "player_shooting.csv", "player_possession.csv",
    "player_passing_type.csv", "player_passing.csv",
    "player_standard_stats.csv"
]

players_path = os.path.join(DATA2_DIR, "players.csv")

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

# Build lookup table indexed by player_name
players_ref = players_df.set_index('player_name')[[
    'country_of_citizenship',   # for nation
    'country_of_birth',          # for country
    'year_of_birth'              # for born
]]

# Process each stats file
for fname in dataset1_files:
    in_path = os.path.join(DATA1_DIR, fname)
    if not os.path.exists(in_path):
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

for fname in dataset1_files:
    path = os.path.join(DATA1_DIR, fname)
    if not os.path.exists(path):
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
    print(f"{fname}: filled {mask.sum()} age values")
    
#build a country → continent dictionary from existing data
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


#fill country & continent; drop rows missing born

players = pd.read_csv(players_path)
players['player_name'] = (players['first_name'].fillna('') + ' ' +
                          players['last_name'].fillna('')).str.strip()
lookup_players = players.set_index('player_name')[['country_of_birth']]

for fname in dataset1_files:
    in_path = os.path.join(DATA1_DIR, fname)
    if not os.path.exists(in_path):
        continue

    df = pd.read_csv(in_path)

    # Merge country_of_birth
    df = df.merge(lookup_players, how='left',
                  left_on='player', right_index=True)

    #  fill COUNTRY 
    before_country_na = df['country'].isna().sum()
    df['country'] = df['country'].fillna(df['country_of_birth'])
    country_filled = before_country_na - df['country'].isna().sum()

    #  fill CONTINENT 
    mask_continent = df['continent'].isna() & df['country'].notna()
    before_continent_na = df['continent'].isna().sum()
    df.loc[mask_continent, 'continent'] = df.loc[mask_continent, 'country'] \
        .map(country_to_continent)
    continent_filled = before_continent_na - df['continent'].isna().sum()

    # Drop helper column
    df.drop(columns=['country_of_birth'], inplace=True)

    #  drop rows still missing born 
    before_rows = len(df)
    df = df.dropna(subset=['born'])
    dropped_rows = before_rows - len(df)

    # Save cleaned file
    out_path = os.path.join(DATA1_DIR, f"{fname}")
    df.to_csv(out_path, index=False)




VAL_FILE = "valuations_with_season_club.csv"
DATA1_DIR = "data/dataset_1"
DATA_DIR = "data"
TOP_N  = 8000
OUT_DIR = "data/global_selected_8000"
os.makedirs(OUT_DIR, exist_ok=True)

# 1 read valuations (all columns) & build a MultiIndex set
val = pd.read_csv(os.path.join(DATA_DIR, VAL_FILE))
val_pairs = set(zip(val['player_name'], val['year']))

# 2 master table of unique (player, season) pairs in order to compute the total number of nulls
master = pd.Series(0, dtype=int,
                   index=pd.MultiIndex(levels=[[], []],
                                       codes=[[], []],
                                       names=['player', 'season']))

for fname in dataset1_files:
    df = pd.read_csv(os.path.join(DATA1_DIR, fname))

    # keep only the rows present in valuations
    mask_val = [(p, s) in val_pairs for p, s in zip(df['player'], df['season'])]
    df = df.loc[mask_val]

    # ensure country & continent present
    df = df[df['country'].notna() & df['continent'].notna()]

    # compute per-row nulls 
    nulls = (df.drop(columns=['player', 'season'])
               .isna()
               .sum(axis=1))

    # index by the pair and sum duplicates
    nulls.index = pd.MultiIndex.from_arrays([df['player'], df['season']],
                                            names=['player', 'season'])
    nulls = nulls.groupby(level=[0, 1]).sum()

    # align and add
    master = master.reindex(master.index.union(nulls.index), fill_value=0)
    master += nulls.reindex(master.index, fill_value=0)

print(f"Unique pairs considered: {len(master):,}")

# 3 choose the best TOP_N unique pairs
top_pairs = (master.sort_values()
                    .head(TOP_N)
                    .index            # MultiIndex
                    .tolist())
pair_set = set(top_pairs)
print(f"Selected exactly {len(pair_set)} unique pairs with minimal nulls")

# 4 export filtered stats files (deduplicated)
for fname in dataset1_files:
    df = pd.read_csv(os.path.join(DATA1_DIR, fname))
    df = df[df.set_index(['player', 'season']).index.isin(pair_set)]
    # drop duplicates per pair
    df = df.drop_duplicates(subset=['player', 'season'], keep='first')
    out_path = os.path.join(OUT_DIR, f"selected_{fname}")
    df.to_csv(out_path, index=False)
    print(f"{fname}: {len(df):,} rows written")

# 5 export filtered valuations file (deduplicated)
val_sel = val[val.set_index(['player_name', 'year']).index.isin(pair_set)]
val_sel = val_sel.drop_duplicates(subset=['player_name', 'year'], keep='first')
val_sel.to_csv(os.path.join(OUT_DIR, "selected_valuations.csv"), index=False)
print(f"Valuations rows written: {len(val_sel):,}")


DATA_DIR   = "data/global_selected_8000"      
PLAYERS_CSV = "data/dataset_2/players.csv"              
OUT_DIR   = "data/global_selected_8000"
os.makedirs(OUT_DIR, exist_ok=True)
STAT_FILES = [f for f in os.listdir(DATA_DIR)
              if f.startswith("selected_cleaned_player_") and f.endswith(".csv")]
VAL_FILE   = "selected_valuations.csv"

# columns to drop from players.csv
DROP_COLS = [
    "first_name", "last_name", "current_club_id", "player_code",
    "date_of_birth", "contract_expiration_date", "agent_name",
    "image_url", "url", "current_club_domestic_competition_id",
    "current_club_name", "market_value_in_eur"
]
# --------------------------------------------------------------------

# 1 build unified player_name in players.csv
players = pd.read_csv(PLAYERS_CSV)
players["player_name"] = (
    players["first_name"].fillna("") + " " + players["last_name"].fillna("")
).str.strip()

# 2 collect every player in any selected file
selected_players = set()

val_df = pd.read_csv(os.path.join(DATA_DIR, VAL_FILE))
selected_players.update(val_df["player_name"].unique())

for f in STAT_FILES:
    tmp = pd.read_csv(os.path.join(DATA_DIR, f), usecols=["player"])
    selected_players.update(tmp["player"].unique())

# 3 build filtered lookup and drop unnecessary cols
player_lookup = (
    players[players["player_name"].isin(selected_players)]
      .drop(columns=[c for c in DROP_COLS if c in players.columns])
      .loc[:, ["player_id", "player_name"]]
)
players_filtered = players[players["player_name"].isin(selected_players)].copy()

# drop the unwanted columns
players_filtered = players_filtered.drop(
    columns=[c for c in DROP_COLS if c in players_filtered.columns]
)
players_filtered = players_filtered.drop(columns=["player_name", "position", "sub_position"])

# move player_id first
cols = players_filtered.columns.tolist()
cols.insert(0, cols.pop(cols.index("player_id")))
players_filtered = players_filtered[cols]

# save
players_filtered.to_csv(os.path.join(OUT_DIR, "selected_players.csv"),
                        index=False)
print("players_filtered.csv written with", len(players_filtered), "rows")

# 4 mapping dictionary
id_map = player_lookup.set_index("player_name")["player_id"].to_dict()

def move_player_id_first(df: pd.DataFrame) -> pd.DataFrame:
    #Return df with player_id as first column. 
    cols = df.columns.tolist()
    if "player_id" in cols:
        cols.insert(0, cols.pop(cols.index("player_id")))
        df = df[cols]
    return df

# 5 ── stats files
for f in STAT_FILES:
    df = pd.read_csv(os.path.join(DATA_DIR, f))
    df["player_id"] = df["player"].map(id_map)
    if "rk" in df.columns:
        df = df.drop(columns="rk")
    df = move_player_id_first(df)
    df.to_csv(os.path.join(OUT_DIR, f), index=False)

val_df["player_id"] = val_df["player_name"].map(id_map)
val_df = move_player_id_first(val_df)
val_df.to_csv(os.path.join(OUT_DIR, VAL_FILE), index=False)


VAL_PATH = "data/global_selected_8000/selected_valuations.csv"

val_df = pd.read_csv(VAL_PATH)

# 2) rename columns
val_df = val_df.rename(columns={
    "player_name": "player",            # player_name → player
    "year": "season",                   # year        → season
    "club_in_year": "club_in_season"    # club_in_year→ club_in_season
})

# 3) put player_id first again 
cols = val_df.columns.tolist()
if "player_id" in cols:
    cols.insert(0, cols.pop(cols.index("player_id")))
    val_df = val_df[cols]

val_df.to_csv(VAL_PATH, index=False)
"""
"""
correct_names_path = "data/dataset_2/players.csv"      # CSV with correct names (with accents)
wrong_names_path   = "data/global_selected_8000/selected_cleaned_player_standard_stats.csv"        # CSV with names missing accents
output_path        = "data/final_selected_8000/player_standard_stats.csv"        # Output file

# --- Load the CSV files ---
correct_df = pd.read_csv(correct_names_path)
correct_df = correct_df.rename(columns={'name':'player'})
wrong_df   = pd.read_csv(wrong_names_path)
wrong_df = wrong_df.rename(columns={'name':'player'})

# --- Sanity check: ensure 'player_id' exists ---
if 'player_id' not in correct_df.columns or 'player_id' not in wrong_df.columns:
    raise ValueError("Both files must contain a 'player_id' column")

name_column = "player"

# --- Merge the correct name using player_id ---
merged = wrong_df.merge(
    correct_df[['player_id', name_column]],
    on='player_id',                                                                                                                                        TO DELETE
    how='left',
    suffixes=('', '_correct')
)

# --- Replace names in wrong_df with correct ones ---
if name_column in wrong_df.columns:
    merged[name_column] = merged[f"{name_column}_correct"].combine_first(merged[name_column])
else:
    merged.rename(columns={f"{name_column}_correct": name_column}, inplace=True)

# --- Drop helper column and save ---
merged.drop(columns=[c for c in merged.columns if c.endswith('_correct')], inplace=True)
merged.to_csv(output_path, index=False)
"""
"""
events_path = "data/game_lineups.csv"            # your original CSV
filtered_path = "data/game_events_lineups.csv"  # output file

# --- Load the dataset ---
df = pd.read_csv(events_path)

# --- Ensure 'date' column is in datetime format ---
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# --- Filter rows where date >= 2018-01-01 and <= 2024-12-31 ---
df_filtered = df[
    (df['date'] >= '2018-01-01') & (df['date'] <= '2024-12-31')
]

# --- Save the filtered file ---
df_filtered.to_csv(filtered_path, index=False)

# --- Input paths ---
players_path = "data/final_selected_8000/players.csv"                # all players (with player_id, player_name)
events_path = "data/game_events_lineups.csv"   # filtered game events file
output_path = "data/game_played.csv"             # output

# --- Load data ---
players_df = pd.read_csv(players_path)
events_df  = pd.read_csv(events_path)

events_df = events_df.rename(columns={'player_name':'player'})

# --- Ensure date is datetime ---
events_df['date'] = pd.to_datetime(events_df['date'], errors='coerce')

# --- Extract season (year) from date ---
events_df['season'] = events_df['date'].dt.year

# --- Keep only seasons between 2018–2024 (safety) ---
events_df = events_df[(events_df['season'] >= 2018) & (events_df['season'] <= 2024)]

# --- Count games per player per season ---
# Assuming each row in events_df represents one game appearance for that player_id
games_per_season = (
    events_df
    .groupby(['player_id', 'season'])
    .size()
    .reset_index(name='played')
)

# --- Merge with players list to ensure all players appear for all seasons ---
# Build a full player-season grid
seasons = range(2018, 2025)
player_season_grid = (
    pd.MultiIndex.from_product([players_df['player_id'], seasons], names=['player_id', 'season'])
    .to_frame(index=False)
)

# --- Join the counts, fill missing games with 0 ---
game_played = (
    player_season_grid
    .merge(games_per_season, on=['player_id', 'season'], how='left')
    .fillna({'played': 0})
)

# --- Add player_name from players.csv ---
game_played = game_played.merge(
    players_df[['player_id', 'player']],
    on='player_id',
    how='left'
)

game_played['played'] = game_played['played'].astype(int)

# --- Reorder columns ---
game_played = game_played[['player_id', 'player', 'season', 'played']]

# --- Save final result ---
game_played.to_csv(output_path, index=False)



players_path = "data/final_selected_8000/players.csv"
events_path  = "data/game_lineups.csv"
games_path   = "data/dataset_2/games.csv"
output_path  = "data/game_played_results.csv"

# ─── LOAD DATA ───────────────────────────────────────────
players_df = pd.read_csv(players_path)
events_df  = pd.read_csv(events_path)
games_df   = pd.read_csv(games_path)

# ─── PARSE DATES & SEASONS ───────────────────────────────
events_df['date'] = pd.to_datetime(events_df['date'], errors='coerce')
events_df['season'] = events_df['date'].dt.year
events_df = events_df[(events_df['season'] >= 2018) & (events_df['season'] <= 2024)]

# ─── MERGE EVENTS WITH GAMES INFO ────────────────────────
# Each event row (player_id, club_id, game_id) now gains home/away goals & IDs
merged = events_df.merge(
    games_df[['game_id', 'home_club_id', 'away_club_id', 'home_club_goals', 'away_club_goals']],
    on='game_id',
    how='left'
)

# ─── FUNCTION TO DETERMINE OUTCOME ───────────────────────
def compute_outcome(row):
    hc, ac = row['home_club_goals'], row['away_club_goals']
    cid = row['club_id']

    # missing info → no contribution
    if pd.isna(hc) or pd.isna(ac) or pd.isna(cid):
        return (0, 0, 0, 0, 0)

    # player club is home
    if cid == row['home_club_id']:
        diff = hc - ac
    # player club is away
    elif cid == row['away_club_id']:
        diff = ac - hc
    else:
        # mismatch, skip
        return (0, 0, 0, 0, 0)

    # classify result
    if diff > 0:
        win = 1
        draw = lose = 0
        win_for_1 = 1 if diff == 1 else 0
        win_for_2plus = 1 if diff >= 2 else 0
    elif diff == 0:
        win = lose = 0
        draw = 1
        win_for_1 = win_for_2plus = 0
    else:
        win = 0
        draw = 0
        lose = 1
        win_for_1 = win_for_2plus = 0

    return (win, draw, lose, win_for_1, win_for_2plus)

# ─── APPLY OUTCOME LOGIC ─────────────────────────────────
merged[['wins', 'draws', 'loses', 'wins_for_1', 'wins_for_2+']] = \
    merged.apply(compute_outcome, axis=1, result_type='expand')

# ─── AGGREGATE PER PLAYER  SEASON ───────────────────────
agg = (
    merged.groupby(['player_id', 'season'], as_index=False)
    .agg({
        'game_id': 'nunique',   # count distinct games played
        'wins': 'sum',
        'draws': 'sum',
        'loses': 'sum',
        'wins_for_1': 'sum',
        'wins_for_2+': 'sum'
    })
    .rename(columns={'game_id': 'games'})
)

# ─── BUILD FULL GRID (PLAYER × SEASON) ────────────────────
seasons = range(2018, 2025)
player_season_grid = (
    pd.MultiIndex.from_product([players_df['player_id'], seasons],
                               names=['player_id', 'season'])
    .to_frame(index=False)
)

# ─── MERGE COUNTS, FILL MISSING WITH 0 ───────────────────
final = (
    player_season_grid
    .merge(agg, on=['player_id', 'season'], how='left')
    .fillna(0)
)

# ─── CAST TO INT ─────────────────────────────────────────
for col in ['games', 'wins', 'draws', 'loses', 'wins_for_1', 'wins_for_2+']:
    final[col] = final[col].astype(int)

# ─── ADD PLAYER NAME ─────────────────────────────────────
final = final.merge(
    players_df[['player_id', 'player']],
    on='player_id',
    how='left'
)

# ─── REORDER COLUMNS ─────────────────────────────────────
final = final[['player_id', 'player', 'season', 'games', 'wins',
               'draws', 'loses', 'wins_for_1', 'wins_for_2+']]

# ─── SAVE RESULT ─────────────────────────────────────────
final.to_csv(output_path, index=False)

valuations_path = "data/final_selected_8000/valuations.csv"
players_path    = "data/final_selected_8000/players.csv"
lineups_path    = "data/game_events_lineups.csv"
clubs_path      = "data/dataset_2/clubs.csv"
output_path     = "data/valuations_with_stadium.csv"

# --- Load datasets ---
valuations = pd.read_csv(valuations_path)
lineups    = pd.read_csv(lineups_path)
clubs      = pd.read_csv(clubs_path)

# --- Ensure date column is datetime ---
lineups['date'] = pd.to_datetime(lineups['date'], errors='coerce')

# --- Extract season from the date ---
lineups['season'] = lineups['date'].dt.year

# --- Filter to seasons 2018–2024 ---
lineups = lineups[(lineups['season'] >= 2018) & (lineups['season'] <= 2024)]

# --- Sort by date and keep first occurrence per player-season ---
lineups = (
    lineups.sort_values('date')
           .drop_duplicates(subset=['player_id', 'season'], keep='first')
)

# --- Merge with clubs to get only stadium name ---
lineups = lineups.merge(
    clubs[['club_id', 'stadium_name']],
    on='club_id',
    how='left'
)

# --- Merge stadium name into valuations ---
valuations = valuations.merge(
    lineups[['player_id', 'season', 'stadium_name']],
    on=['player_id', 'season'],
    how='left'
)

# --- Save result ---
valuations.to_csv(output_path, index=False)

# --- File paths (adjust as needed) ---
players_path  = "data/final_selected_8000/valuations.csv"
transfer_path = "data/dataset_2/transfers.csv"
output_path   = "data/players_with_transfers.csv"

# --- Load datasets ---
players  = pd.read_csv(players_path)
transfer = pd.read_csv(transfer_path)

# --- Ensure date column is datetime ---
transfer['transfer_date'] = pd.to_datetime(transfer['transfer_date'], errors='coerce')

# --- Extract season (year) from the transfer date ---
transfer['season'] = transfer['transfer_date'].dt.year

# --- Keep only relevant columns ---
transfer = transfer[['player_id', 'season', 'transfer_date', 'market_value_in_eur', 'transfer_fee']]

# --- For players with multiple transfers in a season, take the latest one ---
transfer_latest = (
    transfer.sort_values('transfer_date')
            .drop_duplicates(subset=['player_id', 'season'], keep='last')
)

# --- Rename columns for clarity ---
transfer_latest = transfer_latest.rename(columns={
    'transfer_date': 'transfer_date',
    'market_value_in_eur': 'market_value_in_eur',
    'transfer_fee': 'transfer_fee'
})

# --- If players.csv does not yet have season info, extract from the dataset you have ---
# (optional — only if players.csv has per-season rows)
if 'season' not in players.columns:
    print("⚠️ 'players.csv' has no 'season' column. Transfers will be merged only by player_id.")
    enriched = players.merge(transfer_latest.drop(columns='season'),
                             on='player_id', how='left')
else:
    # Merge using both player_id and season
    enriched = players.merge(transfer_latest,
                             on=['player_id', 'season'], how='left')

# --- Save result ---
enriched.to_csv(output_path, index=False)



players_path  = "data/final_selected_8000/valuations.csv"
output_path   = "data/final_selected_8000/valuations.csv"

# --- Load datasets ---
valuations  = pd.read_csv(players_path)
"""
"""
df['market_value_in_eur'] = pd.to_numeric(df['market_value_in_eur'], errors='coerce')


def assign_value_tier(v):
    if pd.isna(v):
        return None
    if v > 80_000_000:
        return "Elite" # Elite (>80M€)
    elif v > 30_000_000:
        return "Top Class" # Top Class (30–80M€)
    elif v > 10_000_000:
        return "Established" #Established (10–30M€)
    elif v > 1_000_000:
        return "Professional" #Professional (1–10M€)
    else:
        return "Emerging " #Emerging (<1M€)

# --- Apply classification ---
df['value_tier_in_eur'] = df['market_value_in_eur'].apply(assign_value_tier)

# --- Save result ---
df.to_csv(output_path, index=False)



# --- Ensure season column is integer ---
df['season'] = pd.to_numeric(df['season'], errors='coerce').astype('Int64')

# --- Compute decade ---
def assign_decade(year):
    if pd.isna(year):
        return None
    decade_start = int(year // 10 * 10)
    return f"{decade_start}s"

df['decade'] = df['season'].apply(assign_decade)

# --- Save result ---
df.to_csv(output_path, index=False)


games_played_path = "data/game_played_results.csv"           # the file to be filtered
games_played = pd.read_csv(games_played_path)
output_games_play ="data/final_selected_8000/game_played.csv"
output_valuations = "data/final_selected_8000/valuations.csv"

valuations['player_id']   = pd.to_numeric(valuations['player_id'], errors='coerce').astype('Int64')
valuations['season']      = pd.to_numeric(valuations['season'], errors='coerce').astype('Int64')
games_played['player_id'] = pd.to_numeric(games_played['player_id'], errors='coerce').astype('Int64')
games_played['season']    = pd.to_numeric(games_played['season'], errors='coerce').astype('Int64')

# Build set of valid pairs from valuations
valid_pairs = set(zip(valuations['player_id'], valuations['season']))

# Keep only matching rows
mask = [(pid, yr) in valid_pairs for pid, yr in zip(games_played['player_id'], games_played['season'])]
games_played_filtered = games_played.loc[mask]

# Save filtered file
games_played_filtered.to_csv(output_games_play, index=False)

if 'club_name' in valuations.columns:
    valuations = valuations.drop(columns=['club_name'])

if 'club_in_season' in valuations.columns:
    valuations = valuations.rename(columns={'club_in_season': 'squad'})

# Save cleaned valuations
valuations.to_csv(output_valuations, index=False)
"""


DATA1_DIR = "data/final_selected_8000"
output_path = "data/postgre"
"""
# List of all stat CSVs
stat_files = [
    "player_defense.csv",
    "player_gca.csv",
    "player_misc.csv",
    "player_passing.csv",
    "player_passing_type.csv",
    "player_possession.csv",
    "player_shooting.csv",
    "player_standard_stats.csv",
    "valuations_with_transfers.csv"
]

# Merge keys common to all
merge_keys = ["player_id", "season"]

merged = None
for file in stat_files:
    path = os.path.join(DATA1_DIR, file)
    df = pd.read_csv(path)
    if merged is None:
        merged = df
    else:
        merged = pd.merge(merged, df, on=merge_keys, how="outer", suffixes=("", "_dup"))

# Drop any duplicate columns from repeated merges
merged = merged.loc[:, ~merged.columns.duplicated()]

# Save the merged dataset
merged_path = os.path.join(DATA1_DIR, "merged_stats_all.csv")
merged.to_csv(merged_path, index=False)

print(f"✅ Merged all stat files → {merged_path}")
"""
"""
# ---- Load base merged file with all attributes (already joined) ----
fact = pd.read_csv(os.path.join(DATA1_DIR, "merged_stats_all.csv"))  # ← or rebuild via merging step

DATA_DIR = output_path
# ---- Load dimensions ----
dim_club     = pd.read_csv(os.path.join(DATA_DIR, "dim_club.csv"))
dim_season   = pd.read_csv(os.path.join(DATA_DIR, "dim_season.csv"))
dim_position = pd.read_csv(os.path.join(DATA_DIR, "dim_position.csv"))
dim_market   = pd.read_csv(os.path.join(DATA_DIR, "dim_market.csv"))
dim_transfer = pd.read_csv(os.path.join(DATA_DIR, "dim_transfer.csv"))
dim_games    = pd.read_csv(os.path.join(DATA_DIR, "dim_games_played.csv"))
dim_age    = pd.read_csv(os.path.join(DATA_DIR, "dim_age.csv"))


# ---- Join IDs from dimensions ----
fact = fact.merge(dim_club[["club_id", "club_name", "competition"]],
                  left_on=["squad", "comp"],
                  right_on=["club_name", "competition"], how="left")
fact = fact.merge(dim_season[["season_id", "season"]], on="season", how="left")
fact = fact.merge(dim_position[["position_id", "position", "general_position"]],
                  on=["position", "general_position"], how="left")
fact = fact.merge(dim_market[["market_id", "market_value_in_eur"]],
                  on="market_value_in_eur", how="left")
fact = fact.merge(dim_transfer[["transfer_id", "player_id", "season"]],
                  on=["player_id", "season"], how="left")
fact = fact.merge(dim_games[["games_id", "player_id", "season"]],
                  on=["player_id", "season"], how="left")
fact = fact.merge(dim_age[["age_id", "age"]],
                  on=["age"], how="left")

# ---- Select only chosen measures ----
measures = [
    "goals", "assists", "xg",
    "progressive_passes", "progressive_carries", "passes_completed", "key_passes",
    "tackles", "interceptions", "clearances", "blocks", "fouls"
    "yellow_cards", "red_cards", "dribblers_challenged",
    "goals_per_90", "assists_per_90", "xg_per_90", "dribblers_tackled",
    "goals_assists", "shot_on_target_per_90", "errors", "aerials_won", "through_balls",
    "carries_into_final_third", "shot_creating_actions", "non_penalty_xg", "shots_per_90"
]

# Keep only columns that exist
measures = [m for m in measures if m in fact.columns]

id_cols = ["player_id", "club_id", "season_id", "position_id",
           "market_id", "transfer_id", "games_id", "age_id"]

fact_table = fact[id_cols + measures].drop_duplicates(subset=id_cols).reset_index(drop=True)

# ---- Add surrogate key and save ----
fact_table.insert(0, "fact_id", range(1, len(fact_table) + 1))
output_path = os.path.join(DATA_DIR, "fact_player_statistics.csv")
fact_table.to_csv(output_path, index=False)

print(f"✅ fact_player_statistics.csv created with {len(fact_table)} rows and {len(fact_table.columns)} columns.")


"""
user = "postgres"
password = "maucione_M03"
host = "localhost"
port = "5432"
database = "Project_DW"

engine = create_engine("postgresql+psycopg2://postgres:maucione_M03@localhost:5432/Football_DW")

DATA_DIR = "data/postgre"
tables = {
    "Dim_Player": "dim_player.csv",
    "Dim_Age": "dim_age.csv",
    "Dim_Position": "dim_position.csv",
    "Dim_Club": "dim_club.csv",
    "Dim_Season": "dim_season.csv",
    "Dim_Market": "dim_market.csv",
    "Dim_Transfer": "dim_transfer.csv",
    "Dim_GamesPlayed": "dim_games_played.csv",
    "Fact_PlayerStats": "fact_player_statistics.csv"
}
"""
for table, filename in tables.items():
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)
    print(f"Uploading {table} ({len(df)} rows)...")
    df.to_sql(table, engine, schema="public", if_exists="append", index=False)
print("✅ All tables uploaded successfully.")



engine = create_engine("postgresql+psycopg2://postgres:maucione_M03@localhost:5432/Football_DW")

df = pd.read_csv("data/final_selected_8000/merged_stats_all.csv",
                 usecols=["player_id", "season", "Fouls", "Yellow_Cards"])

df.to_sql("temp_misc", engine, schema="public", if_exists="replace", index=False)
"""
PATH = "data/postgre/fact_player_statistics.csv"
fact_player_stats = pd.read_csv(PATH)

fact_player_stats['transfer_id'] = pd.to_numeric(fact_player_stats['transfer_id'], errors='coerce').astype('Int64')

fact_player_stats.to_csv(PATH)