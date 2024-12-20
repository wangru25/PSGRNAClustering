# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2024-06-27 13:22:25
LastModifiedBy: Rui Wang
LastEditTime: 2024-07-11 23:29:50
Email: wang.rui@nyu.edu
FilePath: /RNAClustering/src/latex.py
Description: 
'''
# import pandas as pd
# import os, sys

# # Define directories based on vertices_num
# vertices_num = int(sys.argv[1])
# data_collected = sys.argv[2]  # old, new


# if vertices_num == 0: 
#     results_dir = f"../results/{data_collected}"
# else: 
#     results_dir = f"../results/{data_collected}/{vertices_num}_vertices"

# # Load the clustering metrics data
# metrics_df = pd.read_csv(os.path.join(results_dir, 'clustering_metrics.csv'))

# # Select only the required columns
# selected_columns = [
#     'Method', 'Accuracy', 'Cluster 1 Percentage'
# ]
# metrics_df_selected = metrics_df[selected_columns]


# # Convert the dataframe to LaTeX format and save it to a .tex file
# latex_table = metrics_df_selected.to_latex(index=False, column_format='ccccccccccccc', header=True)
# print(latex_table)


# # Save to a .tex file
# latex_file_path = os.path.join(results_dir, 'clustering_metrics_table.tex')
# with open(latex_file_path, 'w') as f:
#     f.write(latex_table)

# print(f"LaTeX table saved to {latex_file_path}")

# # run the code
# # python src/latex.py 6
import numpy as np
import gudhi as gd
import networkx as nx
import matplotlib.pyplot as plt

# Define the distance matrix
distance_matrix = np.array([
    [100, 1, 1, 6, 6, 100],
    [1, 100, 100, 1, 100, 2],
    [1, 100, 100, 100, 1, 100],
    [100, 1, 100, 100, 100, 100],
    [100, 100, 1, 100, 100, 2],
    [100, 2, 100, 100, 2, 100]
])

# distance_matrix = np.array([
#     [100, 1, 1, 100, 100],
#     [1, 100, 100, 1, 2],
#     [1, 100, 100, 1, 100],
#     [100, 1, 1, 100, 2],
#     [100, 2, 100, 2, 100]
# ])

# distance_matrix = np.array([
#     [100, 1, 1, 100, 100],
#     [1, 100, 100, 100, 2],
#     [1, 100, 100, 1, 100],
#     [100, 100, 1, 100, 2],
#     [100, 2, 100, 2, 100]
# ])

# distance_matrix = np.array([
#     [100, 1, 1, 100],
#     [1, 100, 100,  2],
#     [1, 100, 100, 100],
#     [100, 2, 100, 100]
# ])

# Create the Rips complex
rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=4)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

# Function to visualize the simplex tree and print Betti numbers at a given filtration level
def visualize_and_print_betti_at_filtration(simplex_tree, filtration_level):
    G = nx.Graph()
    for simplex, filtration in simplex_tree.get_simplices():
        if filtration <= filtration_level:
            if len(simplex) == 1:
                G.add_node(simplex[0])
            elif len(simplex) == 2:
                G.add_edge(simplex[0], simplex[1])

    # Draw the graph
    pos = nx.spring_layout(G)  # You can change the layout
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=20, font_color="black", font_weight="bold", width=2, edge_color="gray")
    plt.title(f"Simplex Tree Visualization at Filtration {filtration_level}")
    plt.show()

    # Compute and print Betti numbers
    simplex_tree.compute_persistence()
    betti_numbers = simplex_tree.betti_numbers()
    betti_0 = betti_numbers[0] if len(betti_numbers) > 0 else 0
    betti_1 = betti_numbers[1] if len(betti_numbers) > 1 else 0
    print(f"Filtration {filtration_level}: Betti 0 = {betti_0}, Betti 1 = {betti_1}")


# Visualize the simplex tree and print Betti numbers at different filtration levels
for level in [1, 2, 3, 4]:
    visualize_and_print_betti_at_filtration(simplex_tree, level)