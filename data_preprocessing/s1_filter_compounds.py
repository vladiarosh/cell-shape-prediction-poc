"""
initial filtering of the raw file to leave only those compounds that we tested between 8 and 15 times
"""
import pandas as pd
import pyarrow


def filter_compounds(input_file, output_file):
    df = pd.read_parquet(input_file)

    dup_count = df['Metadata_JCP2022'].value_counts()
    valid_compounds = dup_count[(dup_count >= 8) & (dup_count <= 15)].index
    special_condition = df['Metadata_JCP2022'] == 'JCP2022_033924'

    filtered_df = df[df['Metadata_JCP2022'].isin(valid_compounds) | special_condition]

    filtered_df.to_parquet(output_file, index=False)
    return filtered_df
