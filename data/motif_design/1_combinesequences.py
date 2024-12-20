# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2024-11-09 02:59:22
LastModifiedBy: Rui Wang
LastEditTime: 2024-11-17 00:45:23
Email: wang.rui@nyu.edu
FilePath: /RNAClustering/data/motif_design/1_combinesequences.py
Description: 
'''


import os

def combine_files(folder1, folder2, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get lists of files in each folder
    files1 = [f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))]
    files2 = [f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))]

    # Iterate through each combination of files
    for file1 in files1:
        for file2 in files2:
            # Read the content of each file
            with open(os.path.join(folder1, file1), 'r') as f1, open(os.path.join(folder2, file2), 'r') as f2:
                content1 = f1.read().strip()  # Read and strip newlines
                content2 = f2.read().strip()  # Read and strip newlines
            
            # Combine contents in a single line without newlines
            combined_content = content1 + content2

            # Generate the output filename
            output_filename = f"{os.path.splitext(file1)[0]}_{os.path.splitext(file2)[0]}.in"
            output_path = os.path.join(output_folder, output_filename)

            # Write combined content to the output file
            with open(output_path, 'w') as output_file:
                output_file.write('>seq\n')
                output_file.write(combined_content)

            print(f"Combined {file1} and {file2} into {output_filename}")


# Example usage
folder1 = "./2_2"  # Replace with the path to folder 1
folder2 = "./4_20"  # Replace with the path to folder 2
output_folder = "./2_2_4_20"  # Replace with the path to the output folder

combine_files(folder1, folder2, output_folder)
