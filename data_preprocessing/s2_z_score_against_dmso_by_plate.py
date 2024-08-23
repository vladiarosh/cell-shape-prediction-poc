"""
z-score all treatments against the average values of DMSO controls plate-wise
"""
import pandas as pd


def zscore_normalization_by_plate(input_file, output_file):
    df = pd.read_parquet(input_file)

    parameter_columns = df.columns[4:]
    df_dmso = df[df['Metadata_JCP2022'] == 'JCP2022_033924']
    dmso_mean_plate = df_dmso.groupby('Metadata_Plate')[parameter_columns].mean().reset_index()
    dmso_std_plate = df_dmso.groupby('Metadata_Plate')[parameter_columns].std().reset_index()

    df_treatments = df[df['Metadata_JCP2022'] != 'JCP2022_033924']

    df_combined = pd.merge(df_treatments, dmso_mean_plate, on='Metadata_Plate', suffixes=('', '_mean'))
    df_combined = pd.merge(df_combined, dmso_std_plate, on='Metadata_Plate', suffixes=('', '_std'))

    for param in parameter_columns:
        mean_col = f'{param}_mean'
        std_col = f'{param}_std'
        df_combined[param] = (df_combined[param] - df_combined[mean_col]) / df_combined[std_col]

    columns_to_drop = [f'{param}_mean' for param in parameter_columns] + [f'{param}_std' for param in parameter_columns]
    df_combined = df_combined.drop(columns=columns_to_drop)

    df_combined.to_parquet(output_file, index=False)

    return df_combined
