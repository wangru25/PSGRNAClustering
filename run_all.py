# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2024-04-14 22:29:40
LastModifiedBy: Rui Wang
LastEditTime: 2024-12-20 15:32:39
Email: wang.rui@nyu.edu
FilePath: /PSGRNAClustering/run_all.py
Description: 
'''
import os,sys
import pandas as pd
import subprocess


def parse_features(output_str):
    # Strip out the outer brackets and newline, then split by spaces
    cleaned_str = output_str.strip()[2:-2]  # Removes the outermost brackets and newline
    feature_list = [float(x) for x in cleaned_str.split()]
    return feature_list

# existing_IDs_old = pd.read_csv('./data/existing_IDs_old.csv')['IDs'].tolist()
# csv_file_path = './feature/existing_topo_features_old.csv'

existing_IDs_old = pd.read_csv('./data/all_possible_IDs.csv')['IDs'].tolist()
csv_file_path = './feature/all_possible_topo_features.csv'

feature_columns = [f'Feature_{i+1}' for i in list(range(40))]  # Create feature column names
pd.DataFrame(columns=['IDs'] + feature_columns).to_csv(csv_file_path, index=False)

for graph_id in existing_IDs_old:
    print(f"Processing: {graph_id}")
    command = [
        'python', '-m', 'feature',
        '--graph_id', graph_id,
        '--bins', '[0, 1., 2., 3.]',
        '--filtration', '[1., 2., 3.]',
        '--matrices_path', './data/adj_eig'
    ]
    print()
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:  # If the subprocess ran successfully
        feature_data = result.stdout
        feature_values = parse_features(feature_data) # Convert string output to float list
        # Append the feature and ID to the CSV
        pd.DataFrame([[graph_id] + feature_values], columns=['IDs'] + feature_columns).to_csv(csv_file_path, mode='a', header=False, index=False)
    else:
        print(f"Failed to process {graph_id}: {result.stderr}")
