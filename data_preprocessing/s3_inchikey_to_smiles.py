"""
Using PubChem API, I convert InChi ids to SMILES
"""
import pandas as pd
import requests


def get_smiles_from_inchikey(inchikey):
    try:
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey}/property/CanonicalSMILES/JSON'
        print(f"Requesting URL: {url}")  # Debugging statement
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Check the content of the response for debugging
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Content: {response.text[:1000]}")  # Print first 1000 characters

        data = response.json()

        if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
            smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
            return smiles
        else:
            print("Unexpected response format or InChIKey not found.")
            return None
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except KeyError:
        print("Unexpected response format or InChIKey not found.")
        return None


def convert_inchikey_column_to_smiles(input_file, output_file):
    df = pd.read_csv(input_file)
    df['SMILES'] = df['Metadata_InChIKey'].apply(get_smiles_from_inchikey)
    df.to_csv(output_file, index=False)
    return df
