# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2024-05-19 22:46:31
LastModifiedBy: Rui Wang
LastEditTime: 2024-10-14 00:03:38
Email: wang.rui@nyu.edu
FilePath: /RNAClustering/src/get_subgraphs.py
Description: 
'''
import re
import os,sys
import warnings
import pandas as pd
from utils import *

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def readDualGraphs():
    DualGraphsLib=[]
    Graphs=[]
    DualGraphsLib.append(Graphs) # for vertex == 1, no graphs are there
    for i in range(2,10): # will read dual graphs from 2-9 vertices (10 used as the range function stops before the last number)

        Graphs=[]
        file_eigen = "../data/adj_eig/%dEigen"%i
        file_adjMat = "../data/adj_eig/V%dAdjDG"%i
        
        loadEigenvalues(Graphs,i,file_eigen) # load eigen values for dual graphs for vertex number i
        loadAdjMatrices(Graphs,i,file_adjMat) # load adjacency matrices for dual graphs for vertex number i

        DualGraphsLib.append(Graphs)
    
    return DualGraphsLib

def get_Adj(ID):
    
    n = int(ID.split('_')[0])
    Graphs = []
    eigen_file = "../data/adj_eig/%dEigen"%n
    adj_file = "../data/adj_eig/V%dAdjDG"%n
    loadEigenvalues(Graphs,n,eigen_file)
    loadAdjMatrices(Graphs,n,adj_file)
    for g in Graphs:
        if g.graphID == ID:
            A = g.adjMatrix
            return A

def get_Subgraphs(ID):
    
    n = int(ID.split('_')[0])
    A = get_Adj(ID)
    with open(ID+'Adj.txt', 'w') as f:
        for i in range(0,n):    
            for j in range(0,n):   
                f.write(str(A[i][j])+'\t')
            f.write('\n')
    
    os.system('./dualgraph.out -input '+ID+'Adj.txt -len '+str(n)+' -output '+ID+'Sub.txt -all '+ID+'Sub_all.txt')
    
    with open(ID+'Sub_all.txt', 'r') as f:
        lines = f.readlines()
        
    os.system('rm -rf '+ID+'Adj.txt '+ID+'Sub.txt '+ID+'Sub_all.txt')
    
    DualGraphsLib = readDualGraphs()
    subgraphs = []
    
    for l in lines:
        if l[0] == '(':
            edges = [x.strip() for x in l.split('-')]
            numbers = [int(x) for x in re.findall(r"[\w']+",l)]
            vertices =list(set(numbers))
            
            matrix = []
            for a in range(0,len(vertices)):
                tempArray = []
                for b in range(0,len(vertices)):
                    tempArray.append(0)
                matrix.append(tempArray)
            
            for j in range(len(edges)-1): #last entry is empty as there is a "-" at the end of each line
                edge = edges[j]
                indices = [int(x) for x in re.findall(r"[\w']+",edge)] #read each pair (edge). i.e.  (11,3)
                m = vertices.index(indices[0]) #determine the order (index) of the first vertex of the edge. For (11,3) it is 11 and the index is 3 according to vertices
                n = vertices.index(indices[1]) ##determine the order (index) of the first vertex of the edge. It is 3 and the index is 0               
                matrix[m][n]+=1 #increase the number of connections in the adjacency matrix. matrix[3][0] will be increased 1
                matrix[n][m]+=1 #since the matrix is symmetric, increase matrix[0][3] 1.
                
            N=len(vertices)
            if N==1:
                print("1_1\n")
            elif N>9:
                print('Vertex number > 9\n')
            else:
                eigen = calcEigenValues(matrix) # calculate the eigen values for the subgraph matrix
                subgraphID = searchtoAssignID(DualGraphsLib[N-1],0,len(DualGraphsLib[N-1])-1,eigen,matrix)
                subgraphs.append(subgraphID)
    return subgraphs


def is_pseudoknot(ID):
    
    n = int(ID.split('_')[0])
    A = get_Adj(ID)
    with open(ID+'Adj.txt', 'w') as f:
        for i in range(0,n):    
            for j in range(0,n):   
                f.write(str(A[i][j])+'\t')
            f.write('\n')
    
    os.system('./dualgraph.out -input '+ID+'Adj.txt -len '+str(n)+' -output '+ID+'Sub.txt -all '+ID+'Sub_all.txt')
    
    with open(ID+'Sub.txt', 'r') as f:
        lines = f.readlines()
    
    is_pknot = None
    
    for l in lines:
        if 'number of PK blocks:' in l:
            pkn = int(l.split('\n')[0].split(': ')[1])
            if pkn > 0:
                is_pknot = True
            elif pkn == 0:
                is_pknot = False
            else:
                print('Error in finding pseudoknots!')
            break
        
    os.system('rm -rf '+ID+'Adj.txt '+ID+'Sub.txt '+ID+'Sub_all.txt')

    return is_pknot


def get_overlapped_ids(df):
    id_counts = {}
    
    # Count the appearances of each ID in all columns
    for column in df.columns:
        for id in df[column]:
            if id in id_counts:
                id_counts[id] += 1
            else:
                id_counts[id] = 1
    
    # Get lists of IDs appearing in specific number of columns
    ids_in_two_columns = [id for id, count in id_counts.items() if count == 2]
    ids_in_three_columns = [id for id, count in id_counts.items() if count == 3]
    ids_in_four_columns = [id for id, count in id_counts.items() if count == 4]
    ids_in_five_columns = [id for id, count in id_counts.items() if count == 5]

    return ids_in_two_columns, ids_in_three_columns, ids_in_four_columns, ids_in_five_columns
    
if __name__ == '__main__':
    # graph_id = sys.argv[1]
    # subgraphs = get_Subgraphs(graph_id)
    # print(subgraphs)

    # for id in subgraphs:
    #     plotGraph(id)

    vertices_num = int(sys.argv[1])
    data_collected = sys.argv[2]  # old, new

    df_existing_RNA = pd.read_csv('../data/existing_IDs_old.csv')
    
    if vertices_num == 0:
        top_RNAs = pd.read_csv(f'../results/{data_collected}/top_20_cluster_1_samples.csv').head(20)
        top_nonRNAs = pd.read_csv(f'../results/{data_collected}/top_20_cluster_0_samples.csv').head(20)
    else:
        top_RNAs = pd.read_csv(f'../results/{data_collected}/{vertices_num}_vertices/top_20_cluster_1_samples.csv').head(20)
        top_nonRNAs = pd.read_csv(f'../results/{data_collected}/{vertices_num}_vertices/top_20_cluster_0_samples.csv').head(20)

    ids_in_two_columns_RNA, ids_in_three_columns_RNA, ids_in_four_columns_RNA, ids_in_five_columns_RNA = get_overlapped_ids(top_RNAs)
    ids_in_two_columns_nonRNA, ids_in_three_columns_nonRNA, ids_in_four_columns_nonRNA, ids_in_five_columns_nonRNA = get_overlapped_ids(top_nonRNAs)

    all_RNA_ids = ids_in_two_columns_RNA + ids_in_three_columns_RNA + ids_in_four_columns_RNA + ids_in_five_columns_RNA
    all_nonRNA_ids = ids_in_two_columns_nonRNA + ids_in_three_columns_nonRNA + ids_in_four_columns_nonRNA + ids_in_five_columns_nonRNA


    ids_in_five_columns_nonRNA = ['5_39','6_274','6_122','5_87']

    print('graph id & subgraphs & is_subgraph_nonRNA')
    subgraph_list = []
    for graph_id in ids_in_five_columns_nonRNA:
        subgraphs = list(set(get_Subgraphs(graph_id)))
        if subgraphs == ['NA']:
            subgraphs = [f'{graph_id}']
        subgraphs.remove(graph_id)
        vertex_num = int(graph_id.split('_')[0])
        graph_idx = int(graph_id.split('_')[1])
        subgraph_list += subgraphs
    
        if len(subgraphs) == 0:
            print(f'{vertex_num}\_{graph_idx} & / & No \\\\')
        else:
            status = 0
            for idx in subgraphs:
                if idx in all_nonRNA_ids:
                    status += 1
                else:
                    continue
            # latex_str = ', '.join(subgraphs).replace('_', '\_')
            latex_str = ', '.join(['{\color{cyan}' + s.replace('_', '\_') + '}' if s in df_existing_RNA['IDs'].tolist() else s.replace('_', '\_') for s in sorted(subgraphs)])
            # latex_str = ', '.join([s.replace('_', '\_') for s in sorted(subgraphs)])
            if status > 0:
                print(f'{vertex_num}\_{graph_idx} & {latex_str} & Yes \\\\')
            else:
                print(f'{vertex_num}\_{graph_idx} & {latex_str} & No \\\\')

    subgraph_dict = {element: subgraph_list.count(element) for element in set(subgraph_list)}
    print(subgraph_list)
    # print(subgraph_dict)
