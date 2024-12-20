# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2024-04-25 03:03:31
LastModifiedBy: Rui Wang
LastEditTime: 2024-12-11 15:19:02
Email: wang.rui@nyu.edu
FilePath: /RNAClustering/plot.py
Description: 
'''
import os, sys
import pandas as pd
from igraph import *
import matplotlib.pyplot as plt
import src.read_graph as read_graph
import src.get_PHinputs as get_PHinputs
import src.get_PHcomplex as get_PHcomplex
import src.plot_functions as plot_functions
# import src.get_subgraphs as get_subgraphs

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
pio.templates.default = "simple_white"  #'ggplot2', 'seaborn', 'simple_white', 'plotly','plotly_white', 'plotly_dark', 'presentation', 'xgridoff','ygridoff', 'gridon', 'none'


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

def main(graph_id, matrices_path, saving_bar_dir, saving_spectra_dir, saving_filt_graph_dir,saving_graph_dir):
    # load class
    GraphParser = read_graph.GraphParser(matrices_path, graph_id)
    GetPHinputs = get_PHinputs.GetPHInputs()
    GetGudhi = get_PHcomplex.GetGudhi()
    PlotBarcodeSpectra = plot_functions.PlotBarcodeSpectra()
    
    #load graph
    graph_adj = GraphParser.get_graph()
    rips_dis_matrix = GetPHinputs.generate_pts_dis(graph_adj)
    rips_PH = GetGudhi.get_rips_012(rips_dis_matrix, 4)
    filtrations=[0,1,2,3]
    print(f'{graph_id}: \n {graph_adj} \n {rips_dis_matrix} \n {rips_PH}')
    
    # # plot barcode
    # fig, bars_interval = PlotBarcodeSpectra.plot_barcode(rips_PH)
    # if bars_interval > 1:
    #     height = 50 * bars_interval + 410
    # else:
    #     height = 300 * bars_interval + 170
    # pio.write_image(fig, saving_bar_dir, width=1100, height=height)

    # # plot spectra
    # fig_spectra = PlotBarcodeSpectra.plot_spectra(rips_dis_matrix, filtrations=[1,2,3])
    # pio.write_image(fig_spectra, saving_spectra_dir, width=1100, height=500)

    # # plot graphs in different filtration
    # figs = PlotBarcodeSpectra.plot_filtration_graphs(graph_adj, filtrations=filtrations)
    # for i, filt in enumerate(filtrations):
    #     pio.write_image(figs[i], f'{saving_filt_graph_dir}/{graph_id}_{filt}.pdf', width=400, height=400)

    # # plot dual graphs version 1
    # dual_fig= PlotBarcodeSpectra.plot_dualgraph(graph_adj)
    # pio.write_image(dual_fig, f'{saving_graph_dir}/{graph_id}.pdf', width=400, height=400)

    # plot dual graphs version 2
    g, mylayout, bbox = PlotBarcodeSpectra.plot_dualgraph(graph_adj)
    if 'nonRNA-like' in saving_graph_dir:
        plot(g, f'{saving_graph_dir}/{graph_id}.pdf', vertex_color="#999999", vertex_frame_color="#999999", vertex_size=30, edge_color="#999999", edge_width=5, layout=mylayout, rescale=True, bbox=bbox, background="white", margin=75)
    else:
        plot(g, f'{saving_graph_dir}/{graph_id}.pdf', vertex_color="#FD8984", vertex_frame_color="#FD8984", vertex_size=30, edge_color="#FD8984", edge_width=5, layout=mylayout, rescale=True, bbox=bbox, background="white", margin=75)

if __name__ == "__main__":
    vertices_num = int(sys.argv[1])
    data_collected = sys.argv[2]  # old, new
    
    if vertices_num == 0:
        top_RNAs = pd.read_csv(f'./results/{data_collected}/top_20_cluster_1_samples.csv')
        top_nonRNAs = pd.read_csv(f'./results/{data_collected}/top_20_cluster_0_samples.csv')
    else:
        top_RNAs = pd.read_csv(f'./results/{data_collected}/{vertices_num}_vertices/top_20_cluster_1_samples.csv').head(20)
        top_nonRNAs = pd.read_csv(f'./results/{data_collected}/{vertices_num}_vertices/top_20_cluster_0_samples.csv').head(20)

    ids_in_two_columns_RNA, ids_in_three_columns_RNA, ids_in_four_columns_RNA, ids_in_five_columns_RNA = get_overlapped_ids(top_RNAs)
    ids_in_two_columns_nonRNA, ids_in_three_columns_nonRNA, ids_in_four_columns_nonRNA, ids_in_five_columns_nonRNA = get_overlapped_ids(top_nonRNAs)

    all_RNA_ids = ids_in_two_columns_RNA + ids_in_three_columns_RNA + ids_in_four_columns_RNA + ids_in_five_columns_RNA
    all_nonRNA_ids = ids_in_two_columns_nonRNA + ids_in_three_columns_nonRNA + ids_in_four_columns_nonRNA + ids_in_five_columns_nonRNA
    
    # all_RNA_ids = ['4_17']
    # all_RNA_ids = ['6_274']
    # all_RNA_ids = ['4_23']

    for graph_id in all_RNA_ids:
        matrices_path = './data/adj_eig'
        # saving dirs
        saving_bar_dir = f'./img/barcode/{data_collected}/RNA-like/{graph_id}_barcode.pdf'
        saving_spectra_dir = f'./img/spectra/{data_collected}/RNA-like/{graph_id}_spectra.pdf'
        saving_filt_graph_dir = f'./img/filtration_graphs/{data_collected}/RNA-like'
        saving_graph_dir = f'./img/dualgraphs/{data_collected}/RNA-like'
        main(graph_id, matrices_path, saving_bar_dir, saving_spectra_dir, saving_filt_graph_dir, saving_graph_dir)
    
    for graph_id in all_nonRNA_ids:
        matrices_path = './data/adj_eig'
        # saving dirs
        saving_bar_dir = f'./img/barcode/{data_collected}/nonRNA-like/{graph_id}_barcode.pdf'
        saving_spectra_dir = f'./img/spectra/{data_collected}/nonRNA-like/{graph_id}_spectra.pdf'
        saving_filt_graph_dir = f'./img/filtration_graphs/{data_collected}/nonRNA-like'
        saving_graph_dir = f'./img/dualgraphs/{data_collected}/nonRNA-like'
        main(graph_id, matrices_path, saving_bar_dir, saving_spectra_dir, saving_filt_graph_dir, saving_graph_dir)

    # print(f'ids_in_two_columns_RNA: {ids_in_two_columns_RNA}')
    # print(f'ids_in_three_columns_RNA: {ids_in_three_columns_RNA}')
    # print(f'ids_in_four_columns_RNA: {ids_in_four_columns_RNA}')
    # print(f'ids_in_five_columns_RNA: {ids_in_five_columns_RNA}')

    # print('==============================================================')
    # print(f'ids_in_two_columns_nonRNA: {ids_in_two_columns_nonRNA}')
    # print(f'ids_in_three_columns_nonRNA: {ids_in_three_columns_nonRNA}')
    # print(f'ids_in_four_columns_nonRNA: {ids_in_four_columns_nonRNA}')
    # print(f'ids_in_five_columns_nonRNA: {ids_in_five_columns_nonRNA}')
    

    # get subgraphs
    df_topNonRNA = pd.read_csv('../data/predicted_top_ids_cluster_nonRNAlike.csv')
    df_topRNA = pd.read_csv('../data/predicted_top_ids_cluster_RNAlike.csv')
    df_predicted_nonRNA = pd.read_csv('../data/predicted_nonRNA.csv')
    df_existing_RNA = pd.read_csv('../data/existing_IDs_old.csv')

    print('graph id & subgraphs & is_subgraph_nonRNA')
    
    # subgraph_list = []
    # for graph_id in df_topRNA['IDs'].tolist():
    #     subgraphs = list(set(get_subgraphs.get_Subgraphs(graph_id)))
    #     if subgraphs == ['NA']:
    #         subgraphs = [f'{graph_id}']
    #     subgraphs.remove(graph_id)
    #     vertex_num = int(graph_id.split('_')[0])
    #     graph_idx = int(graph_id.split('_')[1])
    #     subgraph_list += subgraphs
    
    #     if len(subgraphs) == 0:
    #         print(f'{vertex_num}\_{graph_idx} & / & No \\\\')
    #     else:
    #         status = 0
    #         for idx in subgraphs:
    #             if idx in df_predicted_nonRNA['IDs'].tolist():
    #                 status += 1
    #             else:
    #                 continue
    #         # latex_str = ', '.join(subgraphs).replace('_', '\_')
    #         latex_str = ', '.join(['{\color{cyan}' + s.replace('_', '\_') + '}' if s in df_existing_RNA['IDs'].tolist() else s.replace('_', '\_') for s in sorted(subgraphs)])
    #         # latex_str = ', '.join([s.replace('_', '\_') for s in sorted(subgraphs)])
    #         if status > 0:
    #             print(f'{vertex_num}\_{graph_idx} & {latex_str} & Yes \\\\')
    #         else:
    #             print(f'{vertex_num}\_{graph_idx} & {latex_str} & No \\\\')
