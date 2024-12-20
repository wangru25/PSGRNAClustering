# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2024-11-09 01:34:02
LastModifiedBy: Rui Wang
LastEditTime: 2024-11-17 00:44:47
Email: wang.rui@nyu.edu
FilePath: /RNAClustering/data/motif_design/0_get_init_design_seq.py
Description: 
'''
import pandas as pd

def extract_sequence_structure(pdb_txt, target_chain, save_file):
    with open(pdb_txt, 'r') as file:
        lines = file.readlines()
    for id, line in enumerate(lines):
        if line.startswith(f"Chain "):
            chain_id = line.replace("Chain ", "").replace(" ", "").strip()
            if chain_id == target_chain:
                sequence = lines[id + 1]
                break
    with open(save_file, 'w') as file:
            file.write(f'{sequence}')
    return sequence

def extract_list_from_file(filename):
    with open(filename, 'r') as file:
        line = file.readline().strip()  
    items = line.split(": ")[1].split(", ")
    cleaned_items = [item.replace("#", "").replace("*", "").replace(",", "") for item in items]
    return cleaned_items


if __name__ == "__main__":
    graph_id = '4_20'
    corresponding_pdbs = extract_list_from_file(f"./{graph_id}/{graph_id}.txt")

    for items in corresponding_pdbs:
        print(items)
        pdb_id = items.split("_")[0]
        target_input = items.split("_")[1]
        pdb_txt = f"/Users/ruiwang/Dropbox/github/Existing-Dual-Search/PDB_DSSR_2D/{pdb_id}.txt"
        # check if pdb_txt exists
        try:
            save_file = f"/Users/ruiwang/Dropbox/Linux_Backup/2023_NYU/1_Training/TamarSchlick/RNAClustering/data/motif_design/{graph_id}/{pdb_id}_{target_input}.in"
            sequence = extract_sequence_structure(pdb_txt, target_input, save_file)
        except:
            print(f"{pdb_txt} does not exist.")