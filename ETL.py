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
""""
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



VAL_FILE = "valuations_with_season_club.csv"
DATA1_DIR = "data/dataset_1"
DATA_DIR = "data"
TOP_N  = 8000
OUT_DIR = "data/global_selected_8000"
os.makedirs(OUT_DIR, exist_ok=True)
# --------------------------------------------------------------------

# 1 ── read valuations (all columns) & build a MultiIndex set
val = pd.read_csv(os.path.join(DATA_DIR, VAL_FILE))
val_pairs = set(zip(val['player_name'], val['year']))

# 2 ── master table of unique (player, season) pairs
master = pd.Series(0, dtype=int,
                   index=pd.MultiIndex(levels=[[], []],
                                       codes=[[], []],
                                       names=['player', 'season']))

for fname in dataset1_files:
    df = pd.read_csv(os.path.join(DATA1_DIR, fname))

    # keep rows present in valuations
    mask_val = [(p, s) in val_pairs for p, s in zip(df['player'], df['season'])]
    df = df.loc[mask_val]

    # ensure country & continent present
    df = df[df['country'].notna() & df['continent'].notna()]

    # compute per-row nulls (exclude keys)
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

# 3 ── choose the best TOP_N unique pairs
top_pairs = (master.sort_values()
                    .head(TOP_N)
                    .index            # MultiIndex
                    .tolist())
pair_set = set(top_pairs)
print(f"Selected exactly {len(pair_set)} unique pairs with minimal nulls")

# 4 ── export filtered stats files (deduplicated)
for fname in dataset1_files:
    df = pd.read_csv(os.path.join(DATA1_DIR, fname))
    df = df[df.set_index(['player', 'season']).index.isin(pair_set)]
    # drop duplicates per pair, keep first
    df = df.drop_duplicates(subset=['player', 'season'], keep='first')
    out_path = os.path.join(OUT_DIR, f"selected_{fname}")
    df.to_csv(out_path, index=False)
    print(f"  ↳ {fname}: {len(df):,} rows written")

# 5 ── export filtered valuations file (deduplicated)
val_sel = val[val.set_index(['player_name', 'year']).index.isin(pair_set)]
val_sel = val_sel.drop_duplicates(subset=['player_name', 'year'], keep='first')
val_sel.to_csv(os.path.join(OUT_DIR, "selected_valuations.csv"), index=False)
print(f"Valuations rows written: {len(val_sel):,}")

print("\n✅ All files in", OUT_DIR,
      "contain the same 8 000 unique player-season pairs with minimal missing data.")
"""
"""
DATA_DIR   = "data/global_selected_8000"      # folder with selected_* CSVs
PLAYERS_CSV = "data/dataset_2/players.csv"              # reference file
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

# 1 ── build unified player_name in players.csv
players = pd.read_csv(PLAYERS_CSV)
players["player_name"] = (
    players["first_name"].fillna("") + " " + players["last_name"].fillna("")
).str.strip()

# 2 ── collect every player in any selected file
selected_players = set()

val_df = pd.read_csv(os.path.join(DATA_DIR, VAL_FILE))
selected_players.update(val_df["player_name"].unique())

for f in STAT_FILES:
    tmp = pd.read_csv(os.path.join(DATA_DIR, f), usecols=["player"])
    selected_players.update(tmp["player"].unique())

# 3 ── build filtered lookup and drop unnecessary cols
player_lookup = (
    players[players["player_name"].isin(selected_players)]
      .drop(columns=[c for c in DROP_COLS if c in players.columns])
      .loc[:, ["player_id", "player_name"]]
)
#player_lookup.to_csv(os.path.join(OUT_DIR, "player_lookup.csv"), index=False)
# --- Save a full reduced players table (OPTIONAL) -------------------
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

# 4 ── mapping dict
id_map = player_lookup.set_index("player_name")["player_id"].to_dict()

def move_player_id_first(df: pd.DataFrame) -> pd.DataFrame:
    """ """Return df with player_id as first column.""" """
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

# 6 ── valuation file
val_df["player_id"] = val_df["player_name"].map(id_map)
val_df = move_player_id_first(val_df)
val_df.to_csv(os.path.join(OUT_DIR, VAL_FILE), index=False)

print("✅ All updated files saved in", OUT_DIR,
      "with player_id as the first column.")"""

VAL_PATH = "data/global_selected_8000/selected_valuations.csv"

# 1) read the file
val_df = pd.read_csv(VAL_PATH)

# 2) rename columns
val_df = val_df.rename(columns={
    "player_name": "player",            # player_name → player
    "year": "season",                   # year        → season
    "club_in_year": "club_in_season"    # club_in_year→ club_in_season
})

# 3) put player_id first again (optional, if you want to keep that rule)
cols = val_df.columns.tolist()
if "player_id" in cols:
    cols.insert(0, cols.pop(cols.index("player_id")))
    val_df = val_df[cols]

# 4) save back (overwrite or new name)
val_df.to_csv(VAL_PATH, index=False)