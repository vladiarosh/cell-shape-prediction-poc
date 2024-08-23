"""
Optional part to test if normalization of latent vectors affects training process
"""
import numpy as np
import pandas as pd
from numpy.linalg import norm


def normalize_features(train_file, val_file, test_file, output_train, output_val, output_test):
    x_train = pd.read_csv(train_file, sep='\t', header=None)
    x_val = pd.read_csv(val_file, sep='\t', header=None)
    x_test = pd.read_csv(test_file, sep='\t', header=None)

    smiles_train = x_train.iloc[:, 0].values
    features_train = x_train.iloc[:, 1:].to_numpy()
    smiles_val = x_val.iloc[:, 0].values
    features_val = x_val.iloc[:, 1:].to_numpy()
    smiles_test = x_test.iloc[:, 0].values
    features_test = x_test.iloc[:, 1:].to_numpy()

    latent_train_normalized = features_train / norm(features_train, axis=1, keepdims=True)
    latent_val_normalized = features_val / norm(features_val, axis=1, keepdims=True)
    latent_test_normalized = features_test / norm(features_test, axis=1, keepdims=True)

    np.savetxt(output_train, np.column_stack((smiles_train, latent_train_normalized)), delimiter='\t', fmt='%s')
    np.savetxt(output_val, np.column_stack((smiles_val, latent_val_normalized)), delimiter='\t', fmt='%s')
    np.savetxt(output_test, np.column_stack((smiles_test, latent_test_normalized)), delimiter='\t', fmt='%s')

    return latent_train_normalized, latent_val_normalized, latent_test_normalized
