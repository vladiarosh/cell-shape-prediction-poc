"""
Averaging the repeats and saving averaged file for next module
Using averaged dataframe to perform random split of the data
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def average_and_split_data(input_file, averaged_output_file, train_output, test_output, val_output):
    # Read the input data
    df = pd.read_parquet(input_file)
    parameter_columns = df.columns[5:]

    # Calculate the average values per SMILES
    average_df = df.groupby('SMILES')[parameter_columns].mean().reset_index()

    # Prepare data for splitting
    compound_ids = average_df['SMILES']
    parameter_columns_averaged = average_df[parameter_columns]

    # Split data into train, validation, and test sets
    x_train, x_temp, y_train, y_temp = train_test_split(compound_ids, parameter_columns_averaged,
                                                        test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # Normalize the y data
    scaler = StandardScaler()
    scaler.fit(y_train)
    y_train_normalized = scaler.transform(y_train)
    y_val_normalized = scaler.transform(y_val)
    y_test_normalized = scaler.transform(y_test)

    # Convert normalized y data to DataFrames
    y_train_normalized_df = pd.DataFrame(y_train_normalized, index=y_train.index, columns=y_train.columns)
    y_val_normalized_df = pd.DataFrame(y_val_normalized, index=y_val.index, columns=y_val.columns)
    y_test_normalized_df = pd.DataFrame(y_test_normalized, index=y_test.index, columns=y_test.columns)

    # Combine SMILES with normalized y data
    train_data = pd.concat([x_train.reset_index(drop=True), y_train_normalized_df.reset_index(drop=True)], axis=1)
    val_data = pd.concat([x_val.reset_index(drop=True), y_val_normalized_df.reset_index(drop=True)], axis=1)
    test_data = pd.concat([x_test.reset_index(drop=True), y_test_normalized_df.reset_index(drop=True)], axis=1)

    # Save combined data to Parquet files
    train_data.to_parquet(train_output, index=False)
    val_data.to_parquet(val_output, index=False)
    test_data.to_parquet(test_output, index=False)

    # Save the averaged data to Parquet file
    average_df.to_parquet(averaged_output_file, index=False)

    # Return the combined data and the averaged data
    return average_df, train_data, val_data, test_data
