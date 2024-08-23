"""
Generation of fingerprints from SMILES
Filtering out the rows that could not be converted to fingerprints
Tanimoto similarity matrix generation
Generation of density matrix, flattening and subsequent hierarchical dendrogram generation
Generation of clusters based on the dendrogram
Stratified data split based on the clusters generated
"""
import pandas as pd
from rdkit import Chem, DataStructs
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.model_selection import train_test_split


def generate_fingerprints_and_similarity(input_file, train_output_file, test_output_file, cutoff_distance):
    df = pd.read_parquet(input_file)
    smiles_list = df['SMILES'].tolist()

    smiles_fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = Chem.RDKFingerprint(mol, fpSize=512, nBitsPerHash=1)
            smiles_fps.append((smiles, fp))

    fingerprints = [item[1] for item in smiles_fps]
    num_fps = len(fingerprints)
    similarity_matrix = np.zeros((num_fps, num_fps))

    for i in range(num_fps):
        for j in range(i, num_fps):
            sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarity_matrix[i][j] = sim
            similarity_matrix[j][i] = sim

    distance_matrix = np.array([[1 - similarity_matrix[i][j] for j in range(num_fps)] for i in range(num_fps)])

    # Convert distance to numpy array for use with scipy and condense the matrix
    distance_matrix = np.array(distance_matrix)
    condensed_distance_matrix = squareform(distance_matrix, checks=False)
    # Create dendrogram
    dendrogram = linkage(condensed_distance_matrix, method='ward')

    cluster_labels = fcluster(dendrogram, t=cutoff_distance, criterion='distance')

    train_indices, test_indices, _, _ = train_test_split(np.arange(num_fps), cluster_labels,
                                                         test_size=0.2, stratify=cluster_labels)
    train_smiles = [smiles_fps[i][0] for i in train_indices]
    test_smiles = [smiles_fps[i][0] for i in test_indices]

    train_df = df[df['SMILES'].isin(train_smiles)].reset_index(drop=True)
    test_df = df[df['SMILES'].isin(test_smiles)].reset_index(drop=True)

    geometrical_columns = train_df.columns[1:]

    # Standardization of data
    scaler = StandardScaler()
    scaler.fit(train_df[geometrical_columns])

    train_df_normalized = train_df.copy()
    train_df_normalized[geometrical_columns] = scaler.transform(train_df[geometrical_columns])
    test_df_normalized = test_df.copy()
    test_df_normalized[geometrical_columns] = scaler.transform(test_df[geometrical_columns])

    train_df_normalized.to_parquet(train_output_file, index=False)
    test_df_normalized.to_parquet(test_output_file, index=False)

    return train_df_normalized, test_df_normalized
