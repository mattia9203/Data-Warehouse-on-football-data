import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 
from functools import reduce
import itertools

base_path = "C:/Users/matti/Desktop/Data-Warehouse-on-football-data/data/"

# All dataset files
file_paths = {
    "player_passing": "player_passing.csv",
    "player_passing_type": "player_passing_type.csv",
    #"player_playing_time": "player_playing_time.csv",
    "player_possession": "player_possession.csv",
    "player_shooting": "player_shooting.csv",
    "player_standard_stats": "player_standard_stats.csv",
    #"valuations": "valuations.csv",
    "player_defense": "player_defense.csv",
    "player_gca": "player_gca.csv",
    #"player_keeper": "player_keeper.csv",
    #"player_keeper_adv": "player_keeper_adv.csv",
    "player_misc": "player_misc.csv"
}

# Common join keys
join_keys = ["player", "season", "squad", "comp"]

# Load and clean datasets
dfs = []
first = True
for name, file in file_paths.items():
    df = pd.read_csv(os.path.join(base_path, file))

    if first:
        dfs.append(df)
        first = False
    else:
        # Remove duplicate columns that would conflict during merge
        cols_to_drop = [col for col in df.columns if col not in join_keys and col in dfs[0].columns]
        df = df.drop(columns=cols_to_drop)
        dfs.append(df)

# Merge all datasets
merged_df = reduce(lambda left, right: pd.merge(left, right, on=join_keys, how="outer"), dfs)

# Report total rows
print(f"\n✅ Total rows after joining all files: {len(merged_df)}")

# Report missing values per column
missing_report = merged_df.isnull().sum().to_frame(name="Missing Values")
missing_report["Missing %"] = (missing_report["Missing Values"] / len(merged_df)) * 100
missing_report = missing_report.sort_values(by="Missing %", ascending=False)

complete_columns = missing_report[missing_report["Missing %"] < 100.0]

# Print them
print(f"\n✅ Columns with 0% missing values ({len(complete_columns)} total):\n")
for col in complete_columns.index.tolist():
    print(f"- {col}",len(complete_columns.index.to_list()))
