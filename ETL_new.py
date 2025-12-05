import pandas as pd
import numpy as np
import os
import re
from rapidfuzz import process
from sqlalchemy import create_engine

# --- CONFIG ---
DATA_DIR = "data"
D2_DIR = os.path.join(DATA_DIR, "dataset_2")
D1_DIR = os.path.join(DATA_DIR, "dataset_1")

# Load reference files
players = pd.read_csv(os.path.join(D2_DIR, "players.csv"))
clubs = pd.read_csv(os.path.join(D2_DIR, "clubs.csv"))
comps = pd.read_csv(os.path.join(D2_DIR, "competitions.csv"))
vals = pd.read_csv(os.path.join(D2_DIR, "player_valuations.csv"))

# 1. Merge basic info (Names, Clubs, Comps) into Valuations
# Join Player Names
vals = vals.merge(players[['player_id', 'first_name', 'last_name']], on='player_id', how='left')
vals['player_name'] = (vals['first_name'].fillna('') + " " + vals['last_name']).str.strip()
vals.drop(columns=['first_name', 'last_name'], inplace=True)

# Join Club Names
vals = vals.merge(clubs[['club_id', 'name']], left_on='current_club_id', right_on='club_id', how='left')
vals.rename(columns={'name': 'club_name'}, inplace=True)

# Join Competition Names
vals = vals.merge(comps[['competition_id', 'name']], left_on='player_club_domestic_competition_id', right_on='competition_id', how='left')
vals.rename(columns={'name': 'competition_name'}, inplace=True)

# Cleanup IDs
vals.drop(columns=['current_club_id', 'player_club_domestic_competition_id', 'club_id', 'competition_id'], inplace=True)

# Fix Dates
vals['date'] = pd.to_datetime(vals['date'], errors='coerce')
vals['year'] = vals['date'].dt.year
vals = vals[vals['year'] >= 2018] # Project scope

print(f"Base valuation set created: {len(vals)} rows")

#---CLEANING DATASET 1 (STATS) ---

# 1. Prepare Reference Data from Players.csv
# We need this to fill gaps in the stats files
p_ref = players.copy()
p_ref['player_name'] = (p_ref['first_name'].fillna('') + ' ' + p_ref['last_name'].fillna('')).str.strip()
p_ref['year_of_birth'] = pd.to_datetime(p_ref['date_of_birth'], errors='coerce').dt.year

# Create a lookup for basic info
ref_lookup = p_ref.set_index('player_name')[['country_of_citizenship', 'country_of_birth', 'year_of_birth']]

# 2. Loop through all stat files to fill Nation, Country, Born
for f in stat_files:
    path = os.path.join(D1_DIR, f)
    if not os.path.exists(path): continue
    
    df = pd.read_csv(path)
    
    # Merge reference info
    df = df.merge(ref_lookup, left_on='player', right_index=True, how='left')
    
    # Fill gaps
    df['nation'] = df['nation'].fillna(df['country_of_citizenship']).fillna(df['country_of_birth'])
    df['country'] = df['country'].fillna(df['country_of_birth'])
    df['born'] = df['born'].fillna(df['year_of_birth'])
    
    # Drop temp columns
    df.drop(columns=['country_of_citizenship', 'country_of_birth', 'year_of_birth'], inplace=True)
    
    # Calculate Age (Season - Born) where missing
    df['born'] = pd.to_numeric(df['born'], errors='coerce')
    df['season'] = pd.to_numeric(df['season'], errors='coerce')
    
    mask_age = df['age'].isna() & df['born'].notna() & df['season'].notna()
    df.loc[mask_age, 'age'] = df.loc[mask_age, 'season'] - df.loc[mask_age, 'born']
    
    # Save overwrite
    df.to_csv(path, index=False)
    print(f"Processed basic info for {f}")

# 3. Handle Continents
# Build a dictionary of Country -> Continent from existing data
country_to_cont = {}
for f in stat_files:
    df = pd.read_csv(os.path.join(D1_DIR, f), usecols=['country', 'continent'])
    existing = df.dropna().set_index('country')['continent'].to_dict()
    country_to_cont.update(existing)

# Apply Continent fill
for f in stat_files:
    path = os.path.join(D1_DIR, f)
    df = pd.read_csv(path)
    
    mask_cont = df['continent'].isna() & df['country'].notna()
    df.loc[mask_cont, 'continent'] = df.loc[mask_cont, 'country'].map(country_to_cont)
    
    # Final cleanup: drop rows where we still don't know when they were born
    df = df.dropna(subset=['born'])
    
    df.to_csv(path, index=False)
    print(f"Filled continents for {f}")

# --- COMPETITION CLEANING ---
# Map weird slugs to proper names
comp_map = {
    'premier-league': 'Premier League',
    'serie-a': 'Serie A',
    'laliga': 'La Liga',
    'bundesliga': 'Bundesliga',
    'ligue-1': 'Ligue 1',
}

vals['competition_name'] = vals['competition_name'].map(comp_map)
vals = vals.dropna(subset=['competition_name']) # Drop leagues outside top 5

# --- CLUB NAME NORMALIZATION ---
# Get list of clubs from the Stats data (Dataset 1) to match against
stat_files = [f for f in os.listdir(D1_DIR) if f.endswith(".csv")]
# Just read one file to get the unique squad list (possession usually has everyone)
temp_df = pd.read_csv(os.path.join(D1_DIR, "player_possession.csv"))

target_clubs = temp_df['squad'].unique()

common_terms = ['Club', 'FC', 'SC', 'Associazione', 'Sportiva', 'De', 'Royal', ]

# 1. Manual Fixes (The "Hard" cases)
manual_club_map = {
    'Stade Rennais Football' : 'Rennes',
    'Manchester United Football' : 'Manchester Utd',
    'Stade brestois 29' : 'Brest',
    'Manchester City Football' : 'Manchester City',
    "Olympique Gymnaste Nice Côte d'Azur" : 'Nice',
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

# 2. Fuzzy Logic for the rest
def clean_and_match(name):
    # Check manual first
    if name in manual_club_map:
        return manual_club_map[name]
    
    # Basic strip
    clean = re.sub(r'\b(?:' + '|'.join(common_terms) + r')\b', '', name, flags=re.IGNORECASE).strip()
    
    # Fuzzy match
    match = process.extractOne(clean, target_clubs)
    if match and match[1] > 85:
        return match[0]
    return name 

vals['club_name'] = vals['club_name'].apply(clean_and_match)

# --- BUILD SEASONAL CLUB LOOKUP ---
# We need to know who played where in each specific year
print("Building Season-Club lookup...")
lookup_list = []

for f in stat_files:
    path = os.path.join(D1_DIR, f)
    if os.path.exists(path):
        df = pd.read_csv(path, usecols=['player', 'season', 'squad'])
        df.rename(columns={'player': 'player_name', 'season': 'year', 'squad': 'club_in_year'}, inplace=True)
        lookup_list.append(df)

# Concatenate and drop duplicates to create a master list of (Player, Year) -> Club
lookup = pd.concat(lookup_list).drop_duplicates()

# Merge into Valuations
vals = vals.merge(lookup, on=['player_name', 'year'], how='left')

# Fallback: If not found in stats, use the 'current' club name from valuations
vals['club_in_year'] = vals['club_in_year'].fillna(vals['club_name'])


# 2. MARKET VALUE RANGES & TIERS
print("Calculating Value Ranges and Tiers...")

def get_tier(v):
    if pd.isna(v): return "Unknown"
    if v > 80000000: return "Elite"
    if v > 30000000: return "Top Class"
    if v > 10000000: return "Established"
    if v > 1000000: return "Professional"
    return "Emerging"

def get_range(val):
    if pd.isna(val): return "Unknown"
    # Logic: 0 -> "0M-10M", 15M -> "10M-20M"
    lower = int(val // 10000000) * 10
    upper = lower + 10
    return f"{lower}M-{upper}M"

vals['value_tier'] = vals['market_value_in_eur'].apply(get_tier)
vals['market_value_range'] = vals['market_value_in_eur'].apply(get_range)
vals['decade'] = (vals['year'] // 10 * 10).astype(str) + "s"

# Save intermediate enriched valuations
vals.to_csv(os.path.join(DATA_DIR, "step4_valuations_enriched.csv"), index=False)


#---GLOBAL FILTERING (TOP 8000) ---

# 1. SCORE PAIRS
# We score every (player, season) based on how many NULLs they have in the stat files
valid_pairs = set(zip(vals['player_name'], vals['year']))
pair_scores = {}

print("Scoring rows based on data quality...")
for fname in stat_files:
    path = os.path.join(D1_DIR, fname)
    if not os.path.exists(path): continue
    
    df = pd.read_csv(path)
    # Filter to only rows that exist in our valuations
    df = df[df.set_index(['player', 'season']).index.isin(valid_pairs)]
    
    # Count nulls per row
    null_counts = df.isnull().sum(axis=1)
    
    for idx, row in df.iterrows():
        key = (row['player'], row['season'])
        pair_scores[key] = pair_scores.get(key, 0) + null_counts[idx]

# 2. SELECT WINNERS
score_df = pd.DataFrame(list(pair_scores.items()), columns=['pair', 'null_score'])
best_pairs = score_df.sort_values('null_score').head(8000)['pair'].tolist()
best_pairs_set = set(best_pairs)

print(f"Selected {len(best_pairs_set)} unique pairs.")
OUT_DIR = "data/global_selected_8000"

# 3. EXPORT CLEANED FILES
# Export Stats
for fname in stat_files:
    path = os.path.join(D1_DIR, fname)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df.set_index(['player', 'season']).index.isin(best_pairs_set)]
        df.drop_duplicates(subset=['player', 'season'], inplace=True)
        df.to_csv(os.path.join(OUT_DIR, f"selected_cleaned_{fname}"), index=False)

# Export Valuations
vals_final = vals[vals.set_index(['player_name', 'year']).index.isin(best_pairs_set)]
vals_final.drop_duplicates(subset=['player_name', 'year'], inplace=True)
vals_final.to_csv(os.path.join(OUT_DIR, "selected_valuations.csv"), index=False)


#---MERGE ALL DATA (MASTER TABLE) ---

# 1. Merge Stats
print("Merging filtered stats...")
merged_stats = None
DIR = "data/global_selected_8000"
files_to_merge = [f for f in os.listdir(DIR) if f.startswith('selected')]

for fname in files_to_merge:
    df = pd.read_csv(os.path.join(DIR, fname))
    if merged_stats is None:
        merged_stats = df
    else:
        merged_stats = merged_stats.merge(df, on=['player', 'season'], how='outer', suffixes=('', '_dup'))

merged_stats = merged_stats.loc[:, ~merged_stats.columns.duplicated()]
merged_stats.rename(columns={'player': 'player_name', 'season': 'year'}, inplace=True)

# 2. Merge with Valuations to create FULL DATA
full_data = merged_stats.merge(vals_final, on=['player_name', 'year'], how='left')

# Save Master Table (useful for debugging)
full_data.to_csv(os.path.join(DATA_DIR, "master_merged_data.csv"), index=False)
print(f"Master dataset created: {len(full_data)} rows.")


#---GENERATE DIMENSIONS FROM MASTER DATA ---

# 1. DIM_MARKET
print("Generating dim_market.csv...")
# Extract unique combinations directly from the data we will load
dim_market = full_data[['market_value_range', 'value_tier']].drop_duplicates().sort_values('market_value_range')
dim_market.reset_index(drop=True, inplace=True)
dim_market.insert(0, 'market_id', range(1, len(dim_market) + 1))
dim_market.rename(columns={'market_value_range': 'Market_Value_Range', 'value_tier': 'Market_Value_Tier'}, inplace=True)
dim_market.to_csv(os.path.join(POSTGRE_DIR, "dim_market.csv"), index=False)


# 2. DIM_CLUB
print("Generating dim_club.csv...")
# Extract unique clubs from the data
dim_club = full_data[['club_in_year', 'competition_name']].drop_duplicates()
dim_club.rename(columns={'club_in_year': 'Club', 'comp': 'Competition'}, inplace=True)
dim_club.reset_index(drop=True, inplace=True)
dim_club.insert(0, 'club_id', range(1, len(dim_club) + 1))

# Add Manual Country/Continent Mapping
print("Mapping Club Countries/Continents manually...")
league_to_country = {
    'Premier League': 'England', 'Serie A': 'Italy', 'La Liga': 'Spain',
    'Bundesliga': 'Germany', 'Ligue 1': 'France'
}
dim_club['Country'] = dim_club['Competition'].map(league_to_country)
dim_club['Continent'] = 'Europe' 
dim_club.to_csv(os.path.join(POSTGRE_DIR, "dim_club.csv"), index=False)


# 3. DIM_SEASON
print("Generating dim_season.csv...")
dim_season = full_data[['year', 'decade']].drop_duplicates().sort_values('year')
dim_season.rename(columns={'year': 'Season', 'decade': 'Decade'}, inplace=True)
dim_season.reset_index(drop=True, inplace=True)
dim_season.insert(0, 'season_id', range(1, len(dim_season) + 1))
dim_season.to_csv(os.path.join(POSTGRE_DIR, "dim_season.csv"), index=False)


# 4. DIM_POSITION
print("Generating dim_position.csv...")
dim_position = full_data[['position', 'general_position']].drop_duplicates()
dim_position.rename(columns={'position': 'Position', 'general_position': 'Role'}, inplace=True)
dim_position.reset_index(drop=True, inplace=True)
dim_position.insert(0, 'position_id', range(1, len(dim_season) + 1))
dim_position.to_csv(os.path.join(POSTGRE_DIR, "dim_position.csv"), index=False)


# 5. DIM_AGE
print("Generating dim_age.csv...")
dim_age = full_data[['age', 'age_range']].drop_duplicates()
dim_age.rename(columns={'age': 'Age', 'age_range': 'Age_Range'}, inplace=True)
dim_age.reset_index(drop=True, inplace=True)
dim_age.insert(0, 'age_id', range(1, len(dim_season) + 1))
dim_age.to_csv(os.path.join(POSTGRE_DIR, "dim_age.csv"), index=False) 


# 6. DIM_PLAYER
print("Generating dim_player.csv...")
# For player, we still need the original players.csv for metadata (Height, Foot, City),
# but we filter it using the IDs present in our Master Table
players_raw = pd.read_csv(os.path.join(D2_DIR, "players.csv"))
active_player_ids = full_data['player_id'].unique()

dim_player = players_raw[players_raw['player_id'].isin(active_player_ids)].copy()

cols_to_keep = ['player_id', 'first_name', 'last_name', 'country_of_birth', 'city_of_birth', 'foot', 'height_in_cm']
existing_cols = [c for c in cols_to_keep if c in dim_player.columns]
dim_player = dim_player[existing_cols].copy()

dim_player['Name'] = (dim_player['first_name'].fillna('') + ' ' + dim_player['last_name'].fillna('')).str.strip()
dim_player.drop(columns=['first_name', 'last_name'], inplace=True, errors='ignore')

dim_player.rename(columns={
    'country_of_birth': 'Country_of_Birth', 'city_of_birth': 'City_of_Birth',
    'foot': 'Foot', 'height_in_cm': 'Height'
}, inplace=True)

# Add Continent map from stats (Dataset 1)
country_to_continent = {}
for fname in stat_files:
    path = os.path.join(D1_DIR, fname)
    if os.path.exists(path):
        try:
            df_geo = pd.read_csv(path, usecols=['country', 'continent']).dropna().drop_duplicates()
            country_to_continent.update(pd.Series(df_geo.continent.values, index=df_geo.country).to_dict())
        except: continue

dim_player['Continent_of_Birth'] = dim_player['Country_of_Birth'].map(country_to_continent).fillna('Unknown')
dim_player.to_csv(os.path.join(POSTGRE_DIR, "dim_player.csv"), index=False)


#--- FINALIZE FACT TABLE ---
print("Mapping IDs to Master Table...")

# Start with the Master Table
fact_final = full_data.copy()

# Join with Dimensions to get IDs
# 1. Market
fact_final = fact_final.merge(dim_market, left_on=['market_value_range', 'value_tier'], right_on=['Market_Value_Range', 'Market_Value_Tier'], how='left')

# 2. Club
fact_final = fact_final.merge(dim_club, left_on=['club_in_year', 'competition_name'], right_on=['Club', 'Competition'], how='left')

# 3. Season
fact_final = fact_final.merge(dim_season, left_on='year', right_on='Season', how='left')

# 4. Position
fact_final = fact_final.merge(dim_pos, left_on='position', right_on='Position', how='left')

# 5. Age
fact_final = fact_final.merge(dim_age, left_on='age', right_on='Age', how='left')

# Select Final Columns
id_cols = ['player_id', 'club_id', 'season_id', 'market_id', 'age_id', 'position_id']

measure_map = {
    'goals': 'Goals', 'goals_per90': 'Goals_per90', 'assists': 'Assists', 'assists_per90': 'Assists_per90',
    'goals_assists': 'Goals_Assists', 'npxg': 'Non_Penalty_xG', 'xg': 'xG', 'xg_per90': 'xG_per90',
    'shots_per90': 'Shots_per90', 'sca': 'Shot_Creating_Actions', 'passes': 'Passes_Completed', 
    'progressive_passes': 'Progressive_Passes', 'progressive_carries': 'Progressive_Carries',
    'key_passes': 'Key_Passes', 'carries_into_final_third': 'Carries_into_Final_Third',
    'through_balls': 'Through_Balls', 'dribbles_tackled': 'Dribblers_Tackled', 
    'dribbles_challenged': 'Dribblers_Challenged', 'tackles': 'Tackles', 'clearances': 'Clearances',
    'interceptions': 'Interceptions', 'blocks': 'Blocks', 'fouls': 'Fouls', 'errors': 'Errors',
    'aerials_won': 'Aerials_Won', 'yellow_cards': 'Yellow_Cards', 'red_cards': 'Red_Cards',
    'market_value_in_eur': 'Market_Value'
}

fact_final.rename(columns=measure_map, inplace=True)
final_cols = id_cols + list(measure_map.values())
existing_cols = [c for c in final_cols if c in fact_final.columns]

fact_table = fact_final[existing_cols].drop_duplicates()
fact_table.insert(0, 'fact_id', range(1, len(fact_table) + 1))

fact_table.to_csv(os.path.join(POSTGRE_DIR, "fact_player_statistics.csv"), index=False)
print("Fact Table created.")


#--- SQL UPLOAD ---

user = "postgres"
password = ""
host = "localhost"
port = "5432"
database = "Football_DW"

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

tables_to_upload = {
    "Dim_Player": "dim_player.csv",
    "Dim_Market": "dim_market.csv",
    "Dim_Club": "dim_club.csv",
    "Dim_Season": "dim_season.csv",
    "Dim_Position": "dim_position.csv",
    "Dim_Age": "dim_age.csv",
    "Fact_PlayerStats": "fact_player_statistics.csv"
}

for tbl, filename in tables_to_upload.items():
    path = os.path.join(POSTGRE_DIR, filename)
    if os.path.exists(path):
        print(f"Uploading {tbl}...")
        df = pd.read_csv(path)
        df.to_sql(tbl, engine, schema="public", if_exists="append", index=False)

print("Process Complete.")


