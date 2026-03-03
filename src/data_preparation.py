import pandas as pd
from pathlib import Path

# Load data
data_dir = Path("data/raw")
df_batch1 = pd.read_csv(data_dir / "batch1_raw.csv")
df_batch2 = pd.read_csv(data_dir / "batch2_raw.csv")

# Add tracking
df_batch1['Source_Batch'] = 'Batch_1'
df_batch2['Source_Batch'] = 'Batch_2'

# Combine
df_combined = pd.concat([df_batch1, df_batch2], ignore_index=True)

# 1. DROP GHOST ROWS: Remove any row where the base Formulation is missing
df_combined = df_combined.dropna(subset=['Formulation']).copy()

# 2. DROP GHOST COLUMNS: Remove any columns pandas named "Unnamed"
df_combined = df_combined.loc[:, ~df_combined.columns.str.contains('^Unnamed')]

# Clean column names
df_combined.columns = df_combined.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('\'', '').str.replace('%', 'pct')

# Clean string columns
df_combined['Formulation'] = df_combined['Formulation'].str.replace("'", "")
df_combined['Dispersent_Type'] = df_combined['Dispersent_Type'].str.replace("'", "")

# Engineer Composite ID
df_combined['Composite_Mix_ID'] = (
    df_combined['Formulation'].astype(str) + "_" + 
    df_combined['Dispersent_Type'].astype(str) + "_" + 
    df_combined['Solid_Content_pct'].astype(str) + "_" + 
    df_combined['Solid_Additive_pct'].astype(str)
)

# Save cleaned data
processed_path = Path("data/processed/combined_slurry_data_cleaned.csv")
df_combined.to_csv(processed_path, index=False)

print(f"Cleaned shape: {df_combined.shape}")
print(f"Total unique physical mixes: {df_combined['Composite_Mix_ID'].nunique()}")
print("\nReplicate Distribution:")
print(df_combined['Composite_Mix_ID'].value_counts().value_counts().sort_index())