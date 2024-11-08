import numpy as np
import pandas as pd
import h5py
from biopandas.pdb import PandasPdb
from scipy.spatial.distance import pdist, squareform
import argparse
import os
from multiprocessing import Pool
import sys
from Bio.PDB import PDBList, PDBParser, PDBIO, Select
import ast
import torch
from functools import partial


mass_dict = {
    "H": 1.007, "C": 12.01, "N": 14.007, "O": 15.999, "S": 32.07,
}

residue_dict = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
                'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19, 'UNK': 20}

processed_proteins = {}

class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        if chain.get_id() == self.chain_id:
            return 1
        else:
            return 0

def get_center_of_mass(atoms):
    atoms = [atom for atom in atoms if atom.element in mass_dict]
    coordinates = np.array([atom.coord for atom in atoms])
    masses = np.array([mass_dict[atom.element] for atom in atoms])
    center_of_mass = np.sum(coordinates * masses[:, None], axis=0) / np.sum(masses)
    return center_of_mass

def download_and_process_pdb(apo_id, chain_id, pdb_dir): #PDBファイルをダウンロードし、指定されたチェーンIDに対応する部分のみを抽出した後、中心座標を計算してPDBファイルに保存する
    pdb_file = os.path.join(pdb_dir, f"pdb{apo_id.lower()}.ent") # このfはformatのf。文字列を作成できる
    if not os.path.isfile(pdb_file):
        pdbl = PDBList()
        pdb_file = pdbl.retrieve_pdb_file(apo_id, pdir=pdb_dir)

    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure(apo_id, pdb_file)
    io = PDBIO()
    io.set_structure(structure)
    io.save(f'{pdb_dir}/{apo_id}_{chain_id}.pdb', ChainSelect(chain_id))
    #error resolving
    print(f"!!!!!!!Keys in structure[0]: {list(structure[0].keys())}")
    ##
    atoms = list(structure[0][chain_id].get_atoms())
    com = get_center_of_mass(atoms)
    for atom in atoms:
        atom.set_coord(atom.get_coord() - com)
    io.save(f'{pdb_dir}/{apo_id}_{chain_id}_centered.pdb', ChainSelect(chain_id))

def calculate_distance_features(coords, com_coords):
    distances = np.linalg.norm(coords - com_coords, axis=1)
    distance_bins = [0, 5, 10, 20, 30, np.inf]
    distance_features = np.zeros((len(coords), len(distance_bins)-1))
    for i in range(len(distance_bins)-1):
        distance_features[:, i] = np.logical_and(distance_bins[i] <= distances, distances < distance_bins[i+1])
    return distance_features

def extract_residues_and_coords(pdb_file, chain_id):
    ppdb = PandasPdb().read_pdb(pdb_file)
    df = ppdb.df['ATOM']
    df = df[df.chain_id == chain_id]
    df_ca = df[df.atom_name == 'CA']
    residues = df_ca.residue_name.to_numpy()
    coords = df_ca[['x_coord', 'y_coord', 'z_coord']].to_numpy()
    return residues, coords

def one_hot_encode(residues):
    one_hot_encoded = np.zeros((len(residues), 21))
    for i, residue in enumerate(residues):
        if residue not in residue_dict:
            raise ValueError(f"Unknown residue: {residue}")
        one_hot_encoded[i, residue_dict[residue]] = 1
    return one_hot_encoded

def construct_graph(coords, threshold=10.0):
    distances = squareform(pdist(coords))
    adjacency_matrix = np.zeros_like(distances, dtype=bool)
    adjacency_matrix[distances <= threshold] = 1
    np.fill_diagonal(adjacency_matrix, 0)
    edge_attr = distances[adjacency_matrix]
    edge_index = np.transpose(np.nonzero(adjacency_matrix))
    return edge_index, edge_attr

def process_protein(protein_path, chain_id, com_coords, apo_id):
    residues, coords = extract_residues_and_coords(protein_path, chain_id)
    com_coords = np.array(ast.literal_eval(com_coords), dtype=np.float64)
    coords = coords - com_coords
    distance_features = calculate_distance_features(coords, com_coords)
    edge_index, edge_attr = construct_graph(coords)
    x = []
    residues = one_hot_encode(residues)
    x.append(residues)
    x.append(coords)
    #embedding = torch.load(f'../../data/seqvec/{apo_id}_{chain_id}.pt')
    #x.append(embedding)
    x = np.concatenate(x, axis=1)
    return x, edge_index, edge_attr

def worker(t, com_df):
    i, (idx, row) = t
    apo_id, chain_id, position_id = row['apo_name'].split('_')
    protein_file = os.path.join(args.pdb_dir, f"{apo_id}_{chain_id}_centered.pdb")
    if not os.path.isfile(protein_file):
        print(f'File {protein_file} does not exist, downloading and processing...')
        download_and_process_pdb(apo_id, chain_id, args.pdb_dir)
    # タンパク質の中心座標の計算
    com_coords = com_df[(com_df['apo_id'] == apo_id) & (com_df['apo_position_id'] == int(position_id)) & (com_df['apo_chain'] == chain_id)]['COM'].values[0]
    x, edge_index, edge_attr = process_protein(protein_file, chain_id, com_coords, apo_id)
    data_type = row['data_type']
    result = {
        'protein_name': row['apo_name'],
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'data_type': data_type,
    }
    return result

# Define the function to save results to hdf5
def save_results_to_hdf5(results, overwrite=False):
    file_names = {'train': 'train_graph_apo_pocket_family.hdf5',  #新しいファイルに変更
                  'validate': 'validate_graph_apo_pocket_family.hdf5', 
                  'test': 'test_graph_apo_pocket_family.hdf5'}
    for result in results:
        protein_name = result['protein_name']
        x = result['x']
        edge_index = result['edge_index']
        edge_attr = result['edge_attr']
        data_type = result['data_type']
        hdf_file_path = os.path.join(args.output_dir, file_names[data_type])
        with h5py.File(hdf_file_path, 'a') as hdf_file:
            if protein_name in hdf_file:
                if overwrite:
                    del hdf_file[protein_name]  # delete the existing group if overwrite is True
                else:
                    continue  # skip this protein if overwrite is False
            grp = hdf_file.create_group(protein_name) # hdf_fileに"grp"グループ（フォルダ）を作成
            # grpグループ内にデータセット（ファイル）を作成
            grp.create_dataset('x', data=x) # 第一引数:保存する際の名前、第二引数:データが保存されているオブジェクト
            grp.create_dataset('edge_index', data=edge_index)
            grp.create_dataset('edge_attr', data=edge_attr)
            # これで以下のようなグループとデータセットが作成される
            #
            # hdf_file_path.h5 --- Group '"protein_name"' (例えば"12ca_A_1)
            #                   |  |
            #                   |  +- Dataset 'x'
            #                   |  |
            #                   |  +- Dataset 'edge_index'
            #                   |  |
            #                   |  +- Dataset 'edge_attr'
            #                   |
            #                  --- Group '"protein_name"' (例えば"12ca_A_2)
            #                      |
            #                      +- Dataset 'x'
            #                      |
            #                      +- Dataset 'edge_index'
            #                      |
            #                      +- Dataset 'edge_attr'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default="input_csv_files/apo_rmsd_value_family.csv") 
    parser.add_argument('--com_file', type=str, default="input_csv_files/pdbbind_analysis_add_COM_DBSCAN.csv")  
    parser.add_argument('--pdb_dir', type=str, default="pdbfile/")                              
    parser.add_argument('--output_dir', type=str, default="graph_files/")                         # 出力結果(hdf5ファイル)を保存するディレクトリ
    parser.add_argument('--overwrite', action='store_true', help="Whether to overwrite data in the HDF5 file if it already exists.") # 上のディレクトリにすでにhdf5ファイルが存在する場合は上書きする
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    com_df = pd.read_csv(args.com_file)
    num_proteins = df.shape[0]

   
    worker_with_com_df = partial(worker, com_df=com_df)
    
    with Pool() as pool:
        results = list(pool.imap(worker_with_com_df, zip(range(num_proteins), df.iterrows())))

    hdf_file_path = os.path.join(args.output_dir, 'train_graph_apo_pocket_family.hdf5')

    save_results_to_hdf5(results, args.overwrite)
    
    # hdf5ファイルの作成ができていることの確認(trainデータを利用)
    if os.path.isfile(hdf_file_path):
        with h5py.File(hdf_file_path, 'r') as hdf_file:
            protein_names = list(hdf_file.keys())
            print(f"First two protein names: {protein_names[:2]}")
            for protein_name in protein_names[:2]:
                print(f"Loading data for protein {protein_name}...")
                x = hdf_file[protein_name]['x'][()]
                edge_index = hdf_file[protein_name]['edge_index'][()]
                edge_attr = hdf_file[protein_name]['edge_attr'][()]
                print(f"Successfully loaded data for protein {protein_name}!")
    else:
        print(f"ファイルが見つかりません: {hdf_file_path}")
    
