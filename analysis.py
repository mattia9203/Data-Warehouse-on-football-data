import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sqlalchemy import create_engine
import os
import statsmodels.formula.api as smf

# Fill in your actual PostgreSQL credentials
user = "postgres"
password = "maucione_M03"
host = "localhost"  # or your IP if it's remote
port = "5432"       # default PostgreSQL port
database = "Football_Data_DW"

# Create SQLAlchemy engine
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")
conn = engine.connect()
print("Connection OK:", conn.closed == False)
conn.close()
os.makedirs('Analysis', exist_ok=True)
"""
#First analysis
def plot_age_value_relationship(df, role_label):
    # Normalize market values to M€
    df['young_mv'] = df['young_mv'] / 1_000_000
    df['older_mv'] = df['older_mv'] / 1_000_000
    
    # Compute gaps
    df['age_gap'] = df['older_age'] - df['young_age']
    df['value_gap'] = df['young_mv'] - df['older_mv']
    
    # Scatter + regression line
    plt.figure(figsize=(8,5))
    sns.regplot(x='age_gap', y='value_gap', data=df, scatter_kws={'alpha':0.7})
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Age Gap (Older - Younger) [years]")
    plt.ylabel("Market Value Gap (Young - Old) [M€]")
    plt.title(f"{role_label}: Age Gap vs Market Value Gap")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"analysis/{role_label.lower()}_scatter.png")
    plt.show()
    plt.close()
    
    # Boxplot: Young vs Old
    df_melted = pd.DataFrame({
        'Young': df['young_mv'],
        'Old':   df['older_mv']
    }).melt(var_name='Player Type', value_name='Market Value (M€)')
    
    plt.figure(figsize=(6,5))
    sns.boxplot(data=df_melted, x='Player Type', y='Market Value (M€)')
    plt.title(f"{role_label}: Market Value Distribution (Young vs Old)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"analysis/{role_label.lower()}_boxplot.png")
    plt.show()
    plt.close()
    
    # Regression summary
    X = sm.add_constant(df['age_gap'])
    y = df['value_gap']
    model = sm.OLS(y, X).fit()
    print(f"📉 {role_label} Regression Results:\n{model.summary()}\n")

# Load and visualize Forwards
df_fwds = pd.read_sql("SELECT * FROM FOOTBALL.vw_age_value_comparison_forward", engine)
plot_age_value_relationship(df_fwds, "Forward")


# Load and visualize Midfielders
df_fwds = pd.read_sql("SELECT * FROM FOOTBALL.vw_age_value_comparison_midfielder", engine)
plot_age_value_relationship(df_fwds, "Midfielder")

# Load and visualize Defenders
df_defs = pd.read_sql("SELECT * FROM FOOTBALL.vw_age_value_comparison_defender", engine)
plot_age_value_relationship(df_defs, "Defender")

#Second analysis 

def metric_importance(view, metrics, role_label):
    # 1. Load data
    df = pd.read_sql(f"SELECT * FROM FOOTBALL.{view}", engine)
    
    # 2. Drop any missing
    df = df.dropna(subset=metrics + ['market_value_m'])
    
    # 3. Compute R² for each metric
    r2_scores = {}
    for m in metrics:
        X = sm.add_constant(df[[m]])
        y = df['market_value_m']
        model = sm.OLS(y, X).fit()
        r2_scores[m] = model.rsquared
        
        # Print the coefficient table
        print(f"--- {role_label}: Metric = {m} ---")
        print(model.summary().tables[1])
        print(f"R² = {model.rsquared:.3f}\n")
    
    # 4. Plot R² bar chart
    r2_series = pd.Series(r2_scores).sort_values(ascending=False)
    plt.figure(figsize=(6,4))
    r2_series.plot(kind='bar')
    plt.title(f"{role_label}: Metric Importance (R²)")
    plt.ylabel("R² (Explained Variance)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"analysis/{role_label.lower()}_r2.png")
    plt.show()

# ——— Forwards ———
forward_metrics = [
    'goals_per_90',
    'assists_per_90',
    'npxg_per90',
    'conv_rate',
    'sca_per90'
]
metric_importance('vw_forward_metric_importance', forward_metrics, 'Forward')


midfielder_metrics = [
    'assists_per_90',
    'progressive_passes_per90',
    'pass_accuracy',
    'sca_per90',
    'tackles_per90'
]
metric_importance('vw_midfielder_metric_importance', midfielder_metrics, 'Midfielder')

# ——— Defenders ———
defender_metrics = [
    'tackles_per90',
    'interceptions_per90',
    'clearances_per90',
    'blocks_per90',
    'pass_accuracy'
]
metric_importance('vw_defender_metric_importance', defender_metrics, 'Defender')

#Third analysis
def analyze_league_bias(view, role_label):
    # 1. Load the view (assumes `engine` is already defined)
    with engine.connect() as conn:
        df = pd.read_sql(f"SELECT * FROM FOOTBALL.{view}", conn)

    # 2. Identify which columns are your five per-90 metrics
    metadata = ['player_name', 'season_text', 'club_name', 'competition', 'market_value_m']
    metrics  = [c for c in df.columns if c not in metadata]

    # 3. Standardize each metric to a z-score: (x – mean)/std
    for m in metrics:
        df[f"{m}_z"] = (df[m] - df[m].mean()) / df[m].std(ddof=0)

    # 4. Composite performance score = sum of the five z-scores
    z_cols = [f"{m}_z" for m in metrics]
    df['perf_score'] = df[z_cols].sum(axis=1)

    # 5. Bin into 5 equal‐sized performance groups (quintiles)
    df['perf_q'] = pd.qcut(df['perf_score'], 5, labels=False)

    # 6. For each quintile, plot market value by league
    for q in sorted(df['perf_q'].unique()):
        subset = df[df['perf_q'] == q]
        top_leagues = subset['competition'].value_counts().nlargest(5).index
        plt.figure(figsize=(10, 5))
        sns.boxplot(
            data=subset[subset['competition'].isin(top_leagues)],
            x='competition', y='market_value_m'
        )
        plt.title(f"{role_label} — Quintile {q+1}: Value by League")
        plt.ylabel("Market Value (M€)")
        plt.xlabel("League")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"analysis/{role_label.lower()}_league_q{q+1}.png")
        plt.close()

    # 7. Regression with league fixed effects
    model = ols('market_value_m ~ perf_score + C(competition)', data=df).fit()
    print(f"\n--- {role_label} League Bias Regression ---")
    print(model.summary2().tables[1])

# Correctly invoke the league-bias analysis:
analyze_league_bias('vw_midfielder_perf_league_bias', 'Midfielder')
analyze_league_bias('vw_forward_perf_league_bias',   'Forward')
analyze_league_bias('vw_defender_perf_league_bias',  'Defender')

#Fourth analysis

def evolution_analysis(view, role_label, conn, out_folder="analysis"):
    #Generates evolution analysis plots and regression for all players or filtered by role.
    # Load data
    df = pd.read_sql(f"SELECT * FROM FOOTBALL.{view}", conn)
    
    # If role-specific, filter
    if role_label != "All Players":
        df = df[df['role'] == role_label]
    
    # Prepare seasons
    df['season_year']  = df['season_year'].astype(int)
    df['season_label'] = df['season_year'].astype(str)
    seasons = sorted(df['season_label'].unique(), key=int)
    df = df.sort_values('season_year')
    
    # A) Boxplot: distribution by season
    plt.figure(figsize=(10,5))
    sns.boxplot(data=df, x='season_label', y='market_value_m', showfliers=False)
    plt.xticks(rotation=45)
    plt.ylabel("Market Value (M€)")
    plt.title(f"{role_label}: Value Distribution by Season")
    plt.tight_layout()
    plt.savefig(f"{out_folder}/{role_label.lower().replace(' ', '_')}_distribution.png")
    plt.close()

    # B) Line plot: average trend by season
    avg = (
        df
        .groupby('season_label')['market_value_m']
        .mean()
        .reindex(seasons)
        .reset_index()
    )
    plt.figure(figsize=(8,5))
    plt.plot(avg['season_label'], avg['market_value_m'], marker='o')
    plt.xticks(rotation=45)
    plt.ylabel("Avg Market Value (M€)")
    plt.title(f"{role_label}: Avg Value by Season")
    plt.tight_layout()
    plt.savefig(f"{out_folder}/{role_label.lower().replace(' ', '_')}_avg_trend.png")
    plt.close()

    # C) Regression: quantify season effects
    model = smf.ols("market_value_m ~ C(season_label)", data=df).fit()
    print(f"\n--- {role_label} Season Effects Regression ---")
    print(model.summary())

conn = engine.connect()
evolution_analysis('vw_market_value_evolution', 'All Players', conn)
for role in ['Midfielder', 'Forward', 'Defender']:
    evolution_analysis('vw_market_value_evolution', role, conn)


#Fifth analysis
def residual_analysis(view, metrics, role_label, engine, top_n=10, out_folder="analysis"):
    # 1. Load data
    df = pd.read_sql(f"SELECT * FROM FOOTBALL.{view}", engine) \
           .dropna(subset=metrics + ['market_value_m'])
    
    # 2. Fit multivariate OLS
    X = sm.add_constant(df[metrics])
    model = sm.OLS(df['market_value_m'], X).fit()
    df['predicted'] = model.predict(X)
    df['residual']  = df['market_value_m'] - df['predicted']
    
    # 3. Identify extremes
    over  = df.nlargest(top_n, 'residual')[['player_name','season_text','residual']]
    under = df.nsmallest(top_n, 'residual')[['player_name','season_text','residual']]
    over['type'], under['type'] = 'Over-valued', 'Under-valued'
    combined = pd.concat([under, over])
    combined['label'] = combined['player_name'] + " (" + combined['season_text'] + ")"
    combined = combined.set_index('label').sort_values('residual')
    
    # 4A. Scatter: Predicted vs Actual
    plt.figure(figsize=(6,6))
    plt.scatter(df['predicted'], df['market_value_m'], alpha=0.5)
    mn, mx = df['predicted'].min(), df['predicted'].max()
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel("Predicted (M€)")
    plt.ylabel("Actual (M€)")
    plt.title(f"{role_label}: Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(f"{out_folder}/{role_label.lower()}_pred_vs_actual.png")
    plt.close()
    
    # 4B. Bar chart: Over/Under-Valued
    plt.figure(figsize=(8,6))
    colors = combined['type'].map({'Over-valued': 'steelblue', 'Under-valued': 'salmon'})
    combined['residual'].plot(kind='barh', color=colors)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.xlabel("Residual (Actual – Predicted, M€)")
    plt.title(f"{role_label}: Top {top_n} Over/Under-valued")
    plt.tight_layout()
    plt.savefig(f"{out_folder}/{role_label.lower()}_residuals_bar.png")
    plt.close()

# Run for each role
residual_analysis('vw_midfielder_multivar',
                  ['assists_per_90','progressive_passes_per90','pass_accuracy','sca_per90','tackles_per90'],
                  'Midfielder',engine)
residual_analysis('vw_forward_multivar',
                  ['goals_per_90','assists_per_90','npxg_per90','conv_rate','sca_per90'],
                  'Forward', engine)
residual_analysis('vw_defender_multivar',
                  ['tackles_per90','interceptions_per90','clearances_per90','blocks_per90','pass_accuracy'],
                  'Defender', engine)
                  

#Sixth analysis
def role_intercept_comparison(views_and_metrics, engine, out_folder="analysis"):
    results = []

    for role_label, (view, metrics) in views_and_metrics.items():
        # Load data
        df = pd.read_sql(f"SELECT * FROM FOOTBALL.{view}", engine)

        # Cast to numeric & drop missing
        for col in metrics + ['market_value_m']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=metrics + ['market_value_m'])

        # If no data, skip
        if df.shape[0] < len(metrics) + 1:
            print(f"⚠️ Skipping {role_label}: not enough data after cleaning ({df.shape[0]} rows).")
            continue

        # Prepare X & y
        X = sm.add_constant(df[metrics])
        y = df['market_value_m']

        # Fit OLS
        model = sm.OLS(y, X).fit()

        # Collect intercept & R²
        results.append({
            'role': role_label,
            'intercept': model.params.get('const', float('nan')),
            'r_squared': model.rsquared
        })

        # Print summary
        print(f"\n--- {role_label} Regression ---")
        print(model.summary())

    # If no results, abort
    if not results:
        print("❌ No valid role models to display.")
        return None

    # Build results DataFrame
    res_df = pd.DataFrame(results).set_index('role')

    # Plot intercepts
    plt.figure(figsize=(6,4))
    res_df['intercept'].plot(kind='bar', color='skyblue')
    plt.ylabel("Baseline Value (M€)")
    plt.title("Baseline Market Value by Role (Intercept)")
    plt.tight_layout()
    plt.savefig(f"{out_folder}/role_intercepts.png")
    plt.show()

    # Plot R²
    plt.figure(figsize=(6,4))
    res_df['r_squared'].plot(kind='bar', color='lightgreen')
    plt.ylabel("R² (Explained Variance)")
    plt.title("Model Fit Quality by Role")
    plt.tight_layout()
    plt.savefig(f"{out_folder}/role_r2.png")
    plt.show()

    return res_df

# 2. Define view-to-metrics mapping
views_and_metrics = {
    'Forward': (
        'vw_forward_position_model',
        ['goals_per_90','assists_per_90','npxg_per90','sca_per90']
    ),
    'Midfielder': (
        'vw_midfielder_position_model',
        ['assists_per_90','progressive_passes_per90','pass_accuracy','sca_per90','tackles_per90']
    ),
    'Defender': (
        'vw_defender_position_model',
        ['tackles_per90','interceptions_per90','clearances_per90','blocks_per90','pass_accuracy']
    )
}

# 3. Execute analysis
res_df = role_intercept_comparison(views_and_metrics, engine)
"""