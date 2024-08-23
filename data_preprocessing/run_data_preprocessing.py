
import os
from s1_filter_compounds import filter_compounds
from s2_z_score_against_dmso_by_plate import zscore_normalization_by_plate
from s3_inchikey_to_smiles import convert_inchikey_column_to_smiles
from s4_merge_smiles_with_filtered import merge_smiles_with_filtered_data
from s5_filter_by_variance import filter_by_variance
from s6_average_and_split import average_and_split_data
from s7_tanimoto_clustering_strat_split import generate_fingerprints_and_similarity
from s8_normalize_latent_vectors import normalize_features


def main(run_step_3: bool, run_step_8: bool, threshold=0.6, cutoff=1.5):
    # Your current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct file paths based on data directory
    data_dir = os.path.join(script_dir, 'data')

    input_files_dir = os.path.join(data_dir, 'input')
    runtime_files_dir = os.path.join(data_dir, 'runtime')

    profiles_file = os.path.join(input_files_dir, 'profiles_var_mad_int.parquet')
    compounds_file = os.path.join(input_files_dir, 'compound.csv')

    filtered_compounds_file = os.path.join(runtime_files_dir, 'filtered_compounds.parquet')
    z_scored_file = os.path.join(runtime_files_dir, 'z_scored_against_dmso.parquet')
    compounds_smiles_file = os.path.join(runtime_files_dir, 'compounds_with_SMILES.csv')
    merged_smiles_file = os.path.join(runtime_files_dir, 'z_scored_against_dmso_with_SMILES.parquet')
    variance_filtered_file = os.path.join(runtime_files_dir, 'variance_filtered_data.parquet')
    averaged_data_file = os.path.join(runtime_files_dir, 'averaged_data.parquet')
    train_file = os.path.join(runtime_files_dir, 'random_split_train_output.parquet')
    val_file = os.path.join(runtime_files_dir, 'random_split_val_output.parquet')
    test_file = os.path.join(runtime_files_dir, 'random_split_test_output.parquet')
    tanimoto_train_file = os.path.join(runtime_files_dir, 'tanimoto_train_output.parquet')
    tanimoto_test_file = os.path.join(runtime_files_dir, 'tanimoto_test_output.parquet')
    latent_train_file = os.path.join(runtime_files_dir, 'latent_train_data_normalized.parquet')
    latent_val_file = os.path.join(runtime_files_dir, 'latent_val_data_normalized.parquet')
    latent_test_file = os.path.join(runtime_files_dir, 'latent_test_data_normalized.parquet')

    # Step 1: Filter compounds taking only ones having between 8 and 15 repeats
    filter_compounds(profiles_file, filtered_compounds_file)

    # Step 2: Z-Score normalization based on DMSO controls (plate-wise)
    zscore_normalization_by_plate(filtered_compounds_file, z_scored_file)

    # Step 3: Convert InChI keys to SMILES using PubChem API (takes a while)
    if run_step_3:
        convert_inchikey_column_to_smiles(compounds_file, compounds_smiles_file)

    # Step 4: Merge SMILES with filtered Data
    merge_smiles_with_filtered_data(compounds_smiles_file, z_scored_file, merged_smiles_file)

    # Step 5: Filter by variance
    filter_by_variance(merged_smiles_file, variance_filtered_file, threshold)

    # Step 6: Averaging the repeats and random split of the data
    average_and_split_data(
        variance_filtered_file,
        averaged_data_file,
        train_file,
        val_file,
        test_file
    )

    # Step 7: Tanimoto similarity matrix based stratified data split
    generate_fingerprints_and_similarity(
        averaged_data_file,
        tanimoto_train_file,
        tanimoto_test_file,
        cutoff
    )

    # Step 8: Normalize latent vectors generated from Nyan encoder (optional)
    # Ensure the files here match the format and naming from previous steps
    if run_step_8:
        normalize_features(
            train_file, val_file, test_file,
            latent_train_file, latent_val_file,
            latent_test_file
        )


if __name__ == "__main__":
    main(False, False)
