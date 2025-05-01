import pandas as pd
import os
from textwrap import indent

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

DATA1_DIR = "data/dataset_1"
dataset1_files = [
    "cleaned_player_defense.csv", "cleaned_player_gca.csv", "cleaned_player_misc.csv",
    "cleaned_player_shooting.csv", "cleaned_player_possession.csv", "cleaned_player_passing_type.csv",
    "cleaned_player_passing.csv", "cleaned_player_standard_stats.csv"
]
DATA2_DIR = "data"
dataset2_files = ["valuations_with_season_club.csv"]

all_files = dataset1_files + dataset2_files

# ─── 2.  helper to pretty-print missing stats  ───────────────────────────────
def report_missing(df, fname, top_n=10):
    total_rows = len(df)
    miss = df.isna().sum()
    pct  = miss.mul(100/total_rows).round(2)
    stats = pd.DataFrame({"missing": miss, "%": pct})
    # keep only columns with at least 1 missing
    stats = stats[stats["missing"] > 0].sort_values("%", ascending=False)

    print(f"\n=== {fname} ===  rows: {total_rows}")
    if stats.empty:
        print("✓ no missing values")
        return
    print("top columns with missing values:")
    print(indent(stats.head(top_n).to_string(), "  "))
    print(f"… total columns with any missing: {len(stats)}")

for fname in all_files:
    if fname == "valuations_with_season_club.csv":
        path = os.path.join(DATA2_DIR, fname)
    else :
        path = os.path.join(DATA1_DIR, fname)
    try:
        df = pd.read_csv(path)
        report_missing(df, fname)
    except FileNotFoundError:
        print(f"  {fname} not found, skipped")

DATA_DIR = "data/global_selected_8000"           # directory where the CSVs live

files_to_check = [
    "selected_cleaned_player_defense.csv",
    "selected_cleaned_player_gca.csv",
    "selected_cleaned_player_misc.csv",
    "selected_cleaned_player_shooting.csv",
    "selected_cleaned_player_possession.csv",
    "selected_cleaned_player_passing_type.csv",
    "selected_cleaned_player_passing.csv",
    "selected_cleaned_player_standard_stats.csv",
    #"valuations_with_season_club.csv"
]
# full paths
files_to_check = [os.path.join(DATA_DIR, f) for f in files_to_check]
# ─────────────────────────────────────────────────────────────────────

OUTPUT_CSV = os.path.join(DATA2_DIR, "missing_value_report.csv")

def first_non_null(series):
    """Return first non-null value of a Series (or '' if none)."""
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else ""

rows = []   # will hold the final report

for path in files_to_check:
    fname = os.path.basename(path)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"⚠️  {fname} not found – skipped")
        continue

    total_rows = len(df)
    for col in df.columns[df.isna().any()]:
        miss_count = df[col].isna().sum()
        rows.append({
            "file": fname,
            "column": col,
            "rows": total_rows,
            "missing": miss_count,
            "%missing": round(miss_count * 100 / total_rows, 2),
            "example_non_null": first_non_null(df[col])
        })

# -------- write the report --------
report_df = pd.DataFrame(rows)
report_df.to_csv(OUTPUT_CSV, index=False)

print("✅ missing-value scan finished")
print(f"📝 report saved to:  {OUTPUT_CSV}")
print(report_df.head())

files_to_check = [
    "selected_cleaned_player_defense.csv",
    "selected_cleaned_player_gca.csv",
    "selected_cleaned_player_misc.csv",
    "selected_cleaned_player_shooting.csv",
    "selected_cleaned_player_possession.csv",
    "selected_cleaned_player_passing_type.csv",
    "selected_cleaned_player_passing.csv",
    "selected_cleaned_player_standard_stats.csv",
    #"valuations_with_season_club.csv"
]

for f in files_to_check:    
    print(f)
    df = pd.read_csv(os.path.join(DATA_DIR, f))
    print("Total missing cells:", df.isna().sum().sum())
    
OUT_DIR = "data/global_selected_8000"        # folder with selected_* CSVs
STAT_FILES = [
    "selected_cleaned_player_defense.csv", "selected_cleaned_player_gca.csv",
    "selected_cleaned_player_misc.csv",    "selected_cleaned_player_shooting.csv",
    "selected_cleaned_player_possession.csv", "selected_cleaned_player_passing_type.csv",
    "selected_cleaned_player_passing.csv",    "selected_cleaned_player_standard_stats.csv", "selected_valuations.csv"
]
VAL_FILE = "selected_valuations.csv"
# --------------------------------------------------------------------
"""
# 1 ── build the reference pair-set from the valuation file
val_df = pd.read_csv(os.path.join(OUT_DIR, VAL_FILE))
ref_pairs = set(zip(val_df["player_name"], val_df["year"]))
assert len(ref_pairs) == 8000, "Valuation file does not have exactly 8 000 pairs!"
print("Reference pair-set size:", len(ref_pairs))

all_ok = True

# 2 ── check every stats file against the reference
for fname in STAT_FILES:
    path = os.path.join(OUT_DIR, fname)
    if not os.path.exists(path):
        print(f"⚠️ {fname} not found – skipped")
        continue

    df = pd.read_csv(path)
    pairs = set(zip(df["player"], df["season"]))

    missing = ref_pairs - pairs      # in reference but not in this file
    extra   = pairs - ref_pairs      # in this file but not in reference

    if not missing and not extra:
        print(f"✅ {fname}: OK – all 8 000 pairs present")
    else:
        all_ok = False
        print(f"❌ {fname}:")
        print(f"   • missing pairs: {len(missing)}")
        print(f"   • extra pairs  : {len(extra)}")
        # uncomment next two lines to print the first few differences
        # if missing: print('     e.g.', list(missing)[:5])
        # if extra:   print('     e.g.', list(extra)[:5])

if all_ok:
    print("\n🎉 Verification passed – every selected file contains the same 8 000 player-season pairs.")
else:
    print("\n⚠️  Verification found inconsistencies. See details above.")
    """

OUTPUT_TXT = os.path.join("data", "all_unique_columns.txt")
# --------------------------------------------------------------------

unique_cols = set()

for fname in STAT_FILES:
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"⚠️  {fname} not found, skipping")
        continue
    df = pd.read_csv(path, nrows=0)          # only header
    unique_cols.update(df.columns)

# sort alphabetically for readability
sorted_cols = sorted(unique_cols)

# save to txt
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    for col in sorted_cols:
        f.write(col + "\n")

print(f"✅ Saved {len(sorted_cols)} unique column names to {OUTPUT_TXT}")