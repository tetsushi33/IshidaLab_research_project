import pandas as pd
import os
from pymol import cmd, stored
import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.Align import PairwiseAligner
from tqdm import tqdm
import sys
import argparse

apo_data_csv = pd.read_csv("../output_csv_files/phaze_1/filtered_apo_protein_info.csv")
holo_data_csv = pd.read_csv("../output_csv_files/phaze_1/holo_protein_info.csv")
apo_holo_pairs_csv = pd.read_csv("../output_csv_files/phaze_1/apo_holo_pairs_sakai.csv")

output_csv_path = f'../output_csv_files/phaze_2/pocket_determined_result.csv'

def main(start_id, end_id):
    processed_id = set() #set型：重複しない要素を持つコレクション

    for protein_id in tqdm(range(start_id, end_id + 1)):
        result = []

        print(f"Start {protein_id}")
        if protein_id in processed_id:
            print(f"Skipping already processed protein_id: {protein_id}")
            continue
        elif protein_id not in apo_data_csv['protein_id'].unique():
            print(f"apo_data don't have {protein_id}")
            continue

        #Pymolセッション開始
        cmd.reinitialize()
        print("=======代表アポタンパク質の決定===============")
        primary_apo = determine_primary_apo(protein_id, apo_holo_pairs_csv, apo_data_csv)