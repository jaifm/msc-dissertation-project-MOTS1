import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

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


def create_reproducible_group_split(
    input_path: Path = Path("data/processed/combined_slurry_data_expanded.csv"),
    output_dir: Path = Path("data/processed/splits"),
    group_column: str = "Composite_Mix_ID",
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Split a processed dataset into reproducible train/test sets.

    The split is performed on the unique values in ``group_column`` so that
    replicates remain in the same partition.
    """

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if group_column not in df.columns:
        raise ValueError(
            f"Required grouping column '{group_column}' was not found in {input_path}."
        )

    unique_groups = df[group_column].dropna().unique()
    train_groups, test_groups = train_test_split(
        unique_groups,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    train_groups = set(train_groups)
    test_groups = set(test_groups)
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(
            "Split leakage detected: the following groups appear in both train and test sets: "
            f"{sorted(overlap)}"
        )

    train_df = df[df[group_column].isin(train_groups)].copy()
    test_df = df[df[group_column].isin(test_groups)].copy()

    train_overlap = set(train_df[group_column].dropna().unique()) & set(test_df[group_column].dropna().unique())
    if train_overlap:
        raise RuntimeError(
            "Split leakage detected after filtering rows: "
            f"{sorted(train_overlap)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "expanded_train.csv"
    test_path = output_dir / "expanded_test.csv"
    manifest_path = output_dir / "expanded_split_manifest.csv"
    metadata_path = output_dir / "expanded_split_metadata.json"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    manifest_df = pd.DataFrame(
        {
            group_column: sorted(train_groups) + sorted(test_groups),
            "split": ["train"] * len(train_groups) + ["test"] * len(test_groups),
        }
    ).sort_values(["split", group_column]).reset_index(drop=True)
    manifest_df.to_csv(manifest_path, index=False)

    metadata = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "manifest_path": str(manifest_path),
        "seed": seed,
        "test_size": test_size,
        "group_column": group_column,
        "n_rows": int(len(df)),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_groups_total": int(len(unique_groups)),
        "n_groups_train": int(len(train_groups)),
        "n_groups_test": int(len(test_groups)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return {
        "train_path": train_path,
        "test_path": test_path,
        "manifest_path": manifest_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }

# ============================================================
# Create paper replication dataset
# ============================================================

df_replication = df_combined.copy()

df_replication = df_replication.drop(columns=["Composite_Mix_ID"])

# Save expanded dataset
replication_path = Path("data/processed/replication_data.csv")
df_replication.to_csv(replication_path, index=False)

print("\n" + "=" * 50)
print("Paper replication dataset created:")
print(f"Shape: {df_replication.shape}")
print(f"Saved to: {replication_path}")
print("Encoding: one-hot (Formulation, Dispersent_Type)")
