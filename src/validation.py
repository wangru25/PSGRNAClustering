# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2024-07-03 16:31:39
LastModifiedBy: Rui Wang
LastEditTime: 2024-07-08 14:42:59
Email: wang.rui@nyu.edu
FilePath: /RNAClustering/src/validation.py
Description: 
'''
import os
import sys
import pandas as pd

def validation_pipeline(IDs_old, IDs_new, cluster_1_samples):
    # Extract IDs
    old_ids = set(IDs_old['IDs'])
    new_ids = set(IDs_new['IDs'])
    cluster_ids = set(cluster_1_samples['IDs'])

    # Initialize lists to store results
    misclassified = []
    regularly_correct = []
    predicted_correct = []
    need_explore = []

    # Classify the IDs based on the given conditions
    for id in cluster_ids:
        if id in old_ids and id not in new_ids:
            misclassified.append(id)
        elif id in old_ids and id in new_ids:
            regularly_correct.append(id)
        elif id not in old_ids and id in new_ids:
            predicted_correct.append(id)
        else:
            need_explore.append(id)

    # Create DataFrames for each category
    misclassified_df = pd.DataFrame(misclassified, columns=['IDs'])
    regularly_correct_df = pd.DataFrame(regularly_correct, columns=['IDs'])
    predicted_correct_df = pd.DataFrame(predicted_correct, columns=['IDs'])
    need_explore_df = pd.DataFrame(need_explore, columns=['IDs'])

    return misclassified_df, regularly_correct_df, predicted_correct_df, need_explore_df

def newly_added_IDs(IDs_old, IDs_new):
    old_ids = set(IDs_old['IDs'])
    new_ids = set(IDs_new['IDs'])
    diff = old_ids - new_ids
    # newly_added = new_ids - old_ids
    return diff

if __name__ == "__main__":
    vertices_num = int(sys.argv[1])
    data_collected = sys.argv[2]  # old, new

    # Load the CSV files
    existing_IDs_old = pd.read_csv('../data/existing_IDs_old.csv'); existing_IDs_old = existing_IDs_old[existing_IDs_old['IDs'].str[:1].astype(int).isin([4, 5,6,7,8,9])]
    existing_IDs_new = pd.read_csv('../data/existing_IDs_new.csv'); existing_IDs_new = existing_IDs_new[existing_IDs_new['IDs'].str[:1].astype(int).isin([4, 5,6,7,8,9])]
    
    clustering_methods = ['KMeans']

    for method in clustering_methods:
        if vertices_num == 45:
            IDs_old = existing_IDs_old[existing_IDs_old['IDs'].str[:1].astype(int).isin([4, 5])]
            IDs_new = existing_IDs_new[existing_IDs_new['IDs'].str[:1].astype(int).isin([4, 5])]
            cluster_1_samples = pd.read_csv(f'../results/{data_collected}/{vertices_num}_vertices/cluster_1_samples_{method}.csv')
        elif vertices_num == 0:
            IDs_old = existing_IDs_old
            IDs_new = existing_IDs_new
            cluster_1_samples = pd.read_csv(f'../results/{data_collected}/cluster_1_samples_{method}.csv')
        else:
            IDs_old = existing_IDs_old[existing_IDs_old['IDs'].str[:1].astype(int).isin([vertices_num])]
            IDs_new = existing_IDs_new[existing_IDs_new['IDs'].str[:1].astype(int).isin([vertices_num])]
            cluster_1_samples = pd.read_csv(f'../results/{data_collected}/{vertices_num}_vertices/cluster_1_samples_{method}.csv')
        
        misclassified_df, regularly_correct_df, predicted_correct_df, need_explore_df = validation_pipeline(IDs_old, IDs_new, cluster_1_samples)
        num_newly_added = newly_added_IDs(existing_IDs_old, existing_IDs_new)
        diff = newly_added_IDs(existing_IDs_old, existing_IDs_new)
        
        print(f"Method: {method}")
        print(f"Misclassified: {misclassified_df.shape[0]}")
        print(f"Regularly Correct: {regularly_correct_df.shape[0]}")
        print(f"Predicted Correct: {predicted_correct_df.shape[0]}")
        print(f"Need Exploration: {need_explore_df.shape[0]}")
        print(f"Newly Added IDs: ", diff)