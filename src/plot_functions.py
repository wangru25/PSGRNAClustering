# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2024-04-25 03:00:44
LastModifiedBy: Rui Wang
LastEditTime: 2024-07-10 23:48:19
Email: wang.rui@nyu.edu
FilePath: /RNAClustering/src/plot_functions.py
Description: 
'''
from igraph import *
import gudhi
# from utils import *
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('tkagg')
import numpy as np
import pandas as pd
import networkx as nx
import plotly.io as pio
import plotly.express as px
from sklearn.manifold import MDS
import plotly.graph_objects as go
pio.templates.default = "simple_white"  #'ggplot2', 'seaborn', 'simple_white', 'plotly','plotly_white', 'plotly_dark', 'presentation', 'xgridoff','ygridoff', 'gridon', 'none'

class PlotBarcodeSpectra:
    def __init__(self):
        pass

    def plot_barcode(self, persistence_intervals):
        '''
        Plot barcode for persistent homology
        '''
        # Set pseudo-death time for infinite values
        pseudo_death = 4  # add an offset

        # Create a mapping for colors
        colors = {0: '#CC6666', 1: '#FFCC99', 2: '#339999'}
        
        # Determine the maximum number of intervals for any dimension
        bars_count = len(persistence_intervals)
        bars_interval = bars_count * 0.1

        # Check whether there is dim = 1 or dim = 2 in the persistence_intervals
        dim1_flag, dim2_flag = False, False
        for dim, _ in persistence_intervals:
            if dim == 1:
                dim1_flag = True
            if dim == 2:
                dim2_flag = True
            if dim1_flag and dim2_flag:
                break

        bars_interval += dim1_flag + dim2_flag

        # Create figure
        fig = go.Figure()

        # Set a counter to increment the y-position of bars to prevent overlapping
        y_counter = dict.fromkeys(range(len(colors)), 0)

        # Add bars for each persistence interval
        for dim, (birth, death) in persistence_intervals:
            if death == float('inf') or death >= pseudo_death:  # Handle infinite death time
                death = pseudo_death
            # Get the color for the current dimension
            color = colors.get(dim, 'black')  # Default to 'black' if the dimension's color is not predefined
            # Determine y-position based on dimension to prevent overlapping
            y_pos = dim + 0.1 * y_counter[dim]  # Slight offset for each bar in the same dimension
            y_counter[dim] += 1  # Increment counter for the dimension
            
            fig.add_trace(go.Scatter(
                x=[birth, death],
                y=[y_pos, y_pos],
                mode='lines',
                line=dict(width=15, color=color),
                name=f'Dimension {dim}'
            ))

        # Update the layout
        fig.update_layout(
            title='',
            xaxis_title='Distance',
            yaxis=dict(
                title='',
                tickvals=list(y_counter.keys()),
                ticktext=[f'Dimension {d}' for d in y_counter.keys()],
            ),
            showlegend=True  # Set to True if you want a legend
        )
        fig.update_layout(font_family="Arial", font=dict(size=30))

        return fig, bars_interval
    
    def plot_spectra(self, rips_dis_matrix, filtrations=[1,2,3]):
            
        data = {'raduis': [], 'min': []}
        for filt_idx, filt in enumerate(filtrations):
            Laplacian = np.zeros((rips_dis_matrix.shape[0], rips_dis_matrix.shape[1]))
            Laplacian[(rips_dis_matrix > 0) & (rips_dis_matrix <= filt)] = -1
            Laplacian += np.diagflat(-np.sum(Laplacian, axis=0))
            eigens = np.sort(np.linalg.eigvalsh(Laplacian))
            eigens_no_zeros = eigens[eigens>1e-5]
            if len(eigens_no_zeros) > 0:
                data['raduis'].append(filt/2)
                data['min'].append(np.min(eigens_no_zeros))
                # data['sum'].append(np.sum(eigens_no_zeros))
                # data['avg'].append(np.mean(eigens_no_zeros))
        df = pd.DataFrame(data)
        print(df)

        fig = px.line(data, x='raduis', y='min', title='Spectra', line_shape='hv')
        for trace in fig.data:
            trace.line.color = '#CC6666'
            trace.line.width = 4  # Increase the line width; adjust as needed
            trace.fill = 'tozeroy'  # Fill the area under the line
        # Set y-axis to start from 0 and dynamically adjust the maximum value based on the 'min' column
        fig.update_xaxes(title_text ='$r$') # add 10% padding
        fig.update_yaxes(title_text ='$\lambda^{r,0}_0$', range=[df['min'].min()-0.1, df['min'].max()+0.1]) # add 10% padding
        fig.update_layout(font_family="Arial", font=dict(size=30))

        return fig
    
    def get_filtration_graphs(self, rips_dis_matrix, filtrations=[0,1,2,3]):
        Gs = []
        for filt_idx, filt in enumerate(filtrations):
            Laplacian = np.zeros((rips_dis_matrix.shape[0], rips_dis_matrix.shape[1]))
            Laplacian[(rips_dis_matrix > 0) & (rips_dis_matrix <= filt)] = -1
            Laplacian += np.diagflat(-np.sum(Laplacian, axis=0))

            D = np.diag(np.sum(Laplacian, axis=1))  # Degree matrix
            A = D - Laplacian  # Adjacency matrix
            # print(A)
            # postive adj in the diagonal
            A = np.where(A < 0, 0, A)

            # Create a graph from the adjacency matrix
            G = nx.convert_matrix.from_numpy_array(A)
            Gs.append(G)
        return Gs

    def networkx_to_plotly(self, G):
        pos = nx.spring_layout(G, seed=7)
        # pos = nx.circular_layout(G)
        # pos = nx.kamada_kawai_layout(G)
        # pos = nx.spectral_layout(G)
        # pos = nx.shell_layout(G)
        # pos = nx.random_layout(G)
        # pos = nx.planar_layout(G)
        # pos = nx.fruchterman_reingold_layout(G, seed=7)
        # pos = nx.bipartite_layout(G, G.nodes())
        # pos = nx.multipartite_layout(G, subset_key='layer')

        # Create Edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Create Nodes
        node_x = [pos[i][0] for i in pos]
        node_y = [pos[i][1] for i in pos]

        # Create Figure
        fig = go.Figure()

        # Add edges as lines
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=3, color='#8BCBDD'),
            hoverinfo='none',
            mode='lines'))

        # Add nodes as scatter
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color='#8BCBDD',
                size=20,
                line_width=2,
                line=dict(color='#349AC8')
                )))
        
        # Update plot layout
        fig.update_layout(
            title='',
            title_font_size=16,
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showline=False, showgrid=False, zeroline=False, showticklabels=False, tickvals=[]),
            yaxis=dict(showline=False, showgrid=False, zeroline=False, showticklabels=False, tickvals=[]),
            annotations = [dict(text="",showarrow=False,xref="paper", yref="paper")],
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    def plot_filtration_graphs(self,rips_dis_matrix, filtrations=[0,1,2,3]):
        figs = []
        Gs = self.get_filtration_graphs(rips_dis_matrix, filtrations=[0,1,2,3])

        for filt_idx, filt in enumerate(filtrations):
            G = Gs[filt_idx]
            fig = self.networkx_to_plotly(G)
            figs.append(fig)
        return figs
    
    # def plot_dualgraphs(self, graph_adj):
    #     G = self.get_dualgraphs(graph_adj)
    #     print(G)
    #     fig = self.networkx_to_plotly(G)
    #     return fig


    def plot_dualgraph(self, graph_adj):
    
        n = int(graph_adj.shape[0])
        A = graph_adj
        g = Graph()
        g.add_vertices(n)

        mylayout=g.layout_circle()
        bbox = BoundingBox(400,400)

        for i in range(0,n):    
            for j in range(i,n):   
                for k in range(0,int(A[i][j])):
                    g.add_edge(i,j)
        if n > 3:
            mylayout = g.layout_fruchterman_reingold()

        return g, mylayout, bbox
        



