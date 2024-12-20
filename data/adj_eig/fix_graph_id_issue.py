# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2024-04-14 22:44:06
LastModifiedBy: Rui Wang
LastEditTime: 2024-04-14 23:08:13
Email: wang.rui@nyu.edu
FilePath: /RNAClustering/data/adj_eig/fix_graph_id_issue.py
Description: 
'''
import pickle
import numpy as np

def fix_id_issue():
    for i in range(2, 10):  # Assuming you might expand this range later
        extracted_data = {i: {}}
        idx = 0
        file_path = f'/Users/rui/Dropbox/Linux_Backup/NYU/1_Training/TamarSchlick/RNAClustering/data/adj_eig/{i}Eigen'
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()  # Remove whitespace and newline characters
                if line.startswith('>'):
                    key = line[1:]  # Remove '>' and take the rest as the key
                    extracted_data[i][key] = f'{i}_{idx + 1}'  # Assign the formatted string to this key
                    idx += 1  # Increment the index for the next key

        # Save the data for the current 'i' using Pickle
        with open(f'index_map_{i}.pkl', 'wb') as pkl_file:
            pickle.dump(extracted_data, pkl_file)

        # print(f"Data for i={i} saved:", extracted_data[i])

# To save data for each 'i':
fix_id_issue()
