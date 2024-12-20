# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2024-07-10 14:41:37
LastModifiedBy: Rui Wang
LastEditTime: 2024-07-11 13:01:40
Email: wang.rui@nyu.edu
FilePath: /RNAClustering/src/latex_acc.py
Description: 
'''
import pandas as pd
import os
import sys

# Define the specific order of vertices_num and their corresponding labels
vertices_labels = {45: 'V4&5', 6: 'V6', 7: 'V7', 8: 'V8', 9: 'V9', 0: 'All'}
vertices_nums = list(vertices_labels.keys())
data_collected = sys.argv[1]  # old, new

# Initialize an empty dictionary to store the data
methods_data = {
    "KMeans": {"Sensitivity": [], "RNA-like": []},
    "MiniBatchKMeans": {"Sensitivity": [], "RNA-like": []},
    "GMM": {"Sensitivity": [], "RNA-like": []},
    "Hierarchical (ward)": {"Sensitivity": [], "RNA-like": []},
    "Spectral": {"Sensitivity": [], "RNA-like": []},
    "Birch": {"Sensitivity": [], "RNA-like": []},
}

# Load the data for each vertices_num and method
for vertices_num in vertices_nums:
    # Define the results directory based on vertices_num
    if vertices_num == 0:
        results_dir = f"../results/{data_collected}"
    else:
        results_dir = f"../results/{data_collected}/{vertices_num}_vertices"

    # Load the clustering metrics data
    metrics_file_path = os.path.join(results_dir, 'clustering_metrics_se.csv')
    if os.path.exists(metrics_file_path):
        metrics_df = pd.read_csv(metrics_file_path)
        
        # Process each method and extract the required metrics
        for method in methods_data.keys():
            method_data = metrics_df[metrics_df['Method'] == method]
            if not method_data.empty:
                sensitivity = method_data['Accuracy'].values[0] * 100
                rna_like = method_data['Cluster 1 Percentage'].values[0] * 100
                sensitivity = f"{sensitivity:.3f}"
                rna_like = f"{rna_like:.3f}"
            else:
                sensitivity, rna_like = 'N/A', 'N/A'
            
            methods_data[method]["Sensitivity"].append(sensitivity)
            methods_data[method]["RNA-like"].append(rna_like)

# Create LaTeX table string
latex_table = """
\\begin{table}[ht!]
    \\centering
    \\setlength\\tabcolsep{4pt}
    \\captionsetup{margin=0.5cm}
    \\caption{Clustering results on Dataset All}
    \\label{tab:prior_metrics}
    \\begin{tabular}{lcccccccccccccccccccccccc}
        \\toprule
        & & \\multicolumn{6}{c}{\\textbf{Datasets}} \\\\
        \\cmidrule(lr){3-8}
        && \\textbf{V4\\&5} & \\textbf{V6} & \\textbf{V7} & \\textbf{V8} & \\textbf{V9} & \\textbf{All} \\\\
        \\midrule
"""

# Add rows for each method
for method, metrics in methods_data.items():
    latex_table += f"        {method} & Sensitivity(\\%)"
    for sensitivity in metrics["Sensitivity"]:
        latex_table += f" & {sensitivity}"
    latex_table += " \\\\\n"
    
    latex_table += f"                  & RNA-like(\\%)"
    for rna_like in metrics["RNA-like"]:
        latex_table += f" & {rna_like}"
    latex_table += " \\\\\n"

latex_table += """        \\bottomrule
    \\end{tabular}
\\end{table}
"""

# Print the LaTeX table
print(latex_table)

# Save to a .tex file
latex_file_path = os.path.join(f"../results/{data_collected}", 'combined_clustering_metrics_table_se.tex')
with open(latex_file_path, 'w') as f:
    f.write(latex_table)

print(f"LaTeX table saved to {latex_file_path}")

# Run the code
# python src/latex.py old
