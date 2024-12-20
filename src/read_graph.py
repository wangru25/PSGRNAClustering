# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2024-04-08 15:30:58
LastModifiedBy: Rui Wang
LastEditTime: 2024-07-11 09:48:25
Email: wang.rui@nyu.edu
FilePath: /RNAClustering/src/read_graph.py
Description: 
'''
import pickle
import numpy as np

class GraphParser:
    def __init__(self, matrices_path, graph_id):
        self.graph_id = graph_id
        self.matrices_path = matrices_path
        self.vertex_num = int(self.graph_id.split('_')[0])

    def get_graph(self):
        with open(f'{self.matrices_path}/index_map_{self.vertex_num}.pkl', 'rb') as pkl_file:
            fixed_idx_dict = pickle.load(pkl_file)
        mapped_graph_id = fixed_idx_dict[self.vertex_num][str(self.graph_id)]
        mapped_id = int(mapped_graph_id.split('_')[1]) - 1
        adj_matrix = np.load(f'{self.matrices_path}/adjancent_{self.vertex_num}.npy')[mapped_id]
        return adj_matrix

    


# # Example usage:
# if __name__ == '__main__':
#     graph_id = '4_29'
#     matrices_path = '../data/adj_eig'
#     GraphParser = GraphParser(matrices_path, graph_id)
#     GraphParser.get_graph()
