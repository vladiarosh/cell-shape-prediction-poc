"""
Variance filtering, threshold can be tweaked in the main file
"""
import pandas as pd


def filter_by_variance(input_file, output_file, variance_threshold):
    df = pd.read_parquet(input_file)
    parameter_columns = df.columns[5:]
    variances = df[parameter_columns].var()

    maximum_variance = max(variances)
    minimum_variance = min(variances)
    print('max_var is', maximum_variance, 'min_var_is', minimum_variance)

    labels_above_threshold = variances[
            variances > minimum_variance + variance_threshold * (maximum_variance - minimum_variance)
        ].index

    final_columns = df.columns[:4].tolist() + labels_above_threshold.tolist()
    df_filtered = df[final_columns]

    df_filtered.to_parquet(output_file, index=False)
    print(f"Shape of the filtered DataFrame: {df_filtered.shape}")

    return df_filtered
