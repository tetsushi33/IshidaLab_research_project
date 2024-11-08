import numpy as np
import pandas as pd
import h5py
from biopandas.mmcif import PandasMmcif
from scipy.spatial.distance import pdist, squareform
import argparse
import os
from multiprocessing import Pool
from Bio.PDB import MMCIFParser, PDBIO, Select
from tqdm import tqdm
from functools import partial 
from Bio.PDB import MMCIFParser
import ast

residue_dict = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19, 'UNK': 20
}

conversion_dict = pd.read_csv('../../input_csv_files/non_amino_2_amino.csv').set_index('Non-standard AA Code')['Standard AA Code'].to_dict()

class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        if chain.get_id() == self.chain_id:
            return 1
        else:
            return 0



def extract_residues_and_coords(mmcif_file, atom_type='CB'):
    """
    Extracts residues and their coordinates from an mmCIF file.
    
    Args:
    - mmcif_file (str): Path to the mmCIF file.
    - atom_type (str): Type of atom to use for coordinates ('CA' or 'CB'). Default is 'CB'.

    Returns:
    - list: List of residue names.
    - numpy.ndarray: Array of coordinates.
    """

    print("-----------------")
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("ID", mmcif_file)
        model = structure[0]
        with open("temp_output.txt", 'w') as file:
            file.write(f"Structure ID: {structure.id}\n")
            for model in structure:
                file.write(f"  Model ID: {model.id}\n")
                for chain in model:
                    file.write(f"    Chain ID: {chain.id}\n")
                    for residue in chain:
                        file.write(f"      Residue: {residue.id} {residue.resname}\n")
            file.write("--------------------------------")
        
        with open("temp_output.txt", 'a') as file:
            file.write(f"Model ID: {model.id}\n")
            for chain in model:
                file.write(f"  Chain ID: {chain.id}\n")
                for residue in chain:
                    file.write(f"    Residue: {residue.id} {residue.resname}\n")



        residues = []
        coords = []

        for chain in model:
            for residue in chain.get_residues():
                if residue.get_id()[0] == ' ':
                    residue_name = residue.get_resname()
                    if residue_name in conversion_dict:
                        residue_name = conversion_dict[residue_name]

                    if atom_type == 'CB' and 'CB' in residue and residue_name != 'GLY': # グリシンは側鎖に水素原子しか持たないためCB原子が存在しない
                        atom = residue['CB']
                    elif 'CA' in residue:
                        atom = residue['CA']
                    else:
                        continue
                    residues.append(residue_name)
                    coords.append(atom.get_coord())

        print(f"Processing file: {mmcif_file}, atom type: {atom_type}")
        if len(coords) == 0:
            print(f"No coordinates extracted for file: {mmcif_file}")
        return residues, np.array(coords)
    except Exception as e:
        print(f"Error processing file {mmcif_file}: {e}")
        raise




def one_hot_encode(residues):
    one_hot_encoded = np.zeros((len(residues), 21))
    for i, residue in enumerate(residues):
        if residue not in residue_dict:
            raise ValueError(f"Unknown residue: {residue}")
        one_hot_encoded[i, residue_dict[residue]] = 1
    return one_hot_encoded

#ランダムエッジするならここ
def construct_graph(coords, threshold=10.0):
    distances = squareform(pdist(coords))
    adjacency_matrix = np.zeros_like(distances, dtype=bool)
    adjacency_matrix[distances <= threshold] = 1
    np.fill_diagonal(adjacency_matrix, 0)
    edge_attr = distances[adjacency_matrix]
    edge_index = np.transpose(np.nonzero(adjacency_matrix))
    return edge_index, edge_attr

# def process_protein(mmcif_path, com_coords, atom_type):
#     residues, coords = extract_residues_and_coords(mmcif_path, atom_type=atom_type)
#     com_coords = np.array(ast.literal_eval(com_coords), dtype=np.float64)
#     coords = coords - com_coords
#     edge_index, edge_attr = construct_graph(coords)
#     x = one_hot_encode(residues)

#     x_with_coords = np.concatenate([x, coords], axis=1)

#     return x_with_coords, edge_index, edge_attr

def process_protein(mmcif_path, com_coords, atom_type):
    try:
        residues, coords = extract_residues_and_coords(mmcif_path, atom_type=atom_type)
        if coords.size == 0:
            raise ValueError(f"No coordinates found for atom type '{atom_type}' in file: {mmcif_path}")

        com_coords = np.array(ast.literal_eval(com_coords), dtype=np.float64)
        if com_coords.shape != (3,):
            raise ValueError(f"Invalid com_coords shape {com_coords.shape} for file: {mmcif_path}")

        coords = coords - com_coords
        edge_index, edge_attr = construct_graph(coords)
        x = one_hot_encode(residues)

        x_with_coords = np.concatenate([x, coords], axis=1)

        return x_with_coords, edge_index, edge_attr
    except Exception as e:
        print(f"Error in process_protein for file {mmcif_path}: {e}")
        raise

def worker(t, df):
    i, (idx, row) = t # 引数tを展開
    try:
        # you can choose COM of C-beta or C-alpha
        atom_type = "CB" # select C-beta 
        # atom_type = "CA" # selct C-alpha
        apo_id = row['apo_name']
        chain_id = row['apo_chain']

        mmcif_file = os.path.join('../../mmcif/apo_center', f"{apo_id}_{chain_id}_centered.cif")
        com_coords = row['pocket_com']
        # print(apo_id, chain_id, com_coords)
        x, edge_index, edge_attr = process_protein(mmcif_file, com_coords, atom_type)
        
        label = row['label']
        data_type = row['data_type']
        result = {
            'protein_name': f"{apo_id}_{chain_id}_{int(row['pocket_id'])}",
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'data_type': data_type,
            'label': label
        }
        return result
    except Exception as e:
        print(f"Error in worker function for index {i}: {e}")
        raise

def save_results_to_hdf5(results, output_dir, overwrite=False):
    file_name = "_graph_data.hdf5"
    # file_name = "_graph_apo_pocket_binary_15_protein_alpha_2.hdf5"
    file_names = {'train': f'train{file_name}', 
                  'validation': f'train{file_name}', 
                  'test': f'test{file_name}'}
    for result in results:
        protein_name = result['protein_name']
        x = result['x']
        edge_index = result['edge_index']
        edge_attr = result['edge_attr']
        label = result['label']
        data_type = result['data_type']
        hdf_file_path = os.path.join(output_dir, file_names[data_type])
        with h5py.File(hdf_file_path, 'a') as hdf_file:
            if protein_name in hdf_file:
                if overwrite:
                    del hdf_file[protein_name]  # existing group is deleted if overwrite is True
                else:
                    continue  # skip this protein if overwrite is False
            grp = hdf_file.create_group(protein_name)
            grp.create_dataset('x', data=x)
            grp.create_dataset('edge_index', data=edge_index)
            grp.create_dataset('edge_attr', data=edge_attr)
            grp.attrs['label'] = label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default="../../input_label_files/protein_family_file_seed-1.csv")
    parser.add_argument('--output_dir', type=str, default="../../hdf5_graph_files/seed-1")
    parser.add_argument('--overwrite', action='store_true', help="Whether to overwrite data in the HDF5 file if it already exists.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    num_proteins = df.shape[0]

    worker_with_df = partial(worker, df=df) # 引数の固定->新たな関数に

    # 並列処理
    with Pool() as pool:
        results = list(tqdm(pool.imap(worker_with_df, zip(range(num_proteins), df.iterrows())), total=num_proteins))

    save_results_to_hdf5(results, args.output_dir, args.overwrite)
