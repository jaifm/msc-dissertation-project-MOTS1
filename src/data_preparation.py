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

# 2. DROP GHOST COLUMNS: Remove any columns pandas named "Unnamed" caused by xlsx to csv conversion.
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

# ============================================================
# Create second dataset with formulation components expanded
# ============================================================

# Formulation composition mapping (wt%)
formulation_components = {
    "F1": {"NMC_pct": 96, "C65_pct": 2, "KS6L_pct": 0, "PVDF_pct": 2},
    "F2": {"NMC_pct": 97, "C65_pct": 1.5, "KS6L_pct": 0, "PVDF_pct": 1.5},
    "F3": {"NMC_pct": 96, "C65_pct": 0, "KS6L_pct": 2, "PVDF_pct": 2},
}

df_expanded = df_combined.copy()

# Map formulation to component columns
for col in ["NMC_pct", "C65_pct", "KS6L_pct", "PVDF_pct"]:
    df_expanded[col] = df_expanded["Formulation"].map(lambda f: formulation_components.get(f, {}).get(col, None))

# Drop original Formulation column (now redundant)
df_expanded = df_expanded.drop(columns=["Formulation"])

# Save expanded dataset
expanded_path = Path("data/processed/combined_slurry_data_expanded.csv")
df_expanded.to_csv(expanded_path, index=False)

print("\n" + "=" * 50)
print("Expanded dataset with formulation components:")
print(f"Shape: {df_expanded.shape}")
print(f"New columns: NMC_pct, C65_pct, KS6L_pct, PVDF_pct")
print(f"Saved to: {expanded_path}")