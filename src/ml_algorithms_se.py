# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2024-07-11 12:49:51
LastModifiedBy: Rui Wang
LastEditTime: 2024-07-11 13:03:45
Email: wang.rui@nyu.edu
FilePath: /RNAClustering/src/ml_algorithms_se.py
Description: 
'''
import os
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
import plotly.graph_objects as go

warnings.filterwarnings('ignore')
pio.templates.default = "simple_white"

def load_data(vertices_num, data_collected):
    data = pd.read_csv(f'../feature/all_possible_features_labels_{data_collected}_version2.csv')
    
    if vertices_num == 0: 
        results_dir = f"../results/{data_collected}"
        imgs_dir = f"../img/clustering/{data_collected}"
        # data_filtered = data
        # print(data_filtered.shape)
        data_filtered = data[data['IDs'].str[:1].astype(int).isin([4,5,6,7,8,9])]# Only use the vertices >=4 data
        print(data_filtered.shape)
    elif vertices_num == 45: # all 4 and 5 vertices
        results_dir = f"../results/{data_collected}/{vertices_num}_vertices"
        imgs_dir = f"../img/clustering/{data_collected}/{vertices_num}_vertices"
        data_filtered = data[data['IDs'].str[:1].astype(int).isin([4,5])]# Only use the 4 and 5 vertices data
    else:
        results_dir = f"../results/{data_collected}/{vertices_num}_vertices"
        imgs_dir = f"../img/clustering/{data_collected}/{vertices_num}_vertices"
        data_filtered = data[data['IDs'].str.startswith(f'{vertices_num}_')]
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)
    
    return data_filtered, results_dir, imgs_dir

def preprocess_data(data):
    if vertices_num == 4 or vertices_num == 5:
        indices = [i for i in np.arange(1,32)]
    else:
        indices = [2, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    # selected_features = [f'Feature_{i}' for i in indices] 
    selected_features = ['S']+['E']
    X = data[selected_features].values

    y = data['labels'].values
    ids = data['IDs'].values

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, ids

def define_clustering_methods():
    rng = np.random.RandomState(seed=1209)
    clustering_methods = {
        "KMeans": KMeans(n_clusters=2, random_state=42),
        "MiniBatchKMeans": MiniBatchKMeans(n_clusters=2, random_state=42),
        "GMM": GaussianMixture(n_components=2, random_state=42),
        # "Hierarchical (ward)": AgglomerativeClustering(n_clusters=2, linkage='ward'),
        "Spectral": SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42),
        "Birch": Birch(n_clusters=2)
    }
    return clustering_methods

def evaluate_clustering(X, y, ids, clustering_methods):
    top_20_cluster_1 = {}
    top_20_cluster_0 = {}
    cluster_1_samples_dict = {}
    cluster_0_samples_dict = {}
    metrics = []

    for name, method in clustering_methods.items():
        try:
            if name == "GMM":
                labels = method.fit_predict(X)
            else:
                labels = method.fit(X).labels_

            # Assign clusters: cluster with most RNA existing structures (label 1) as cluster 1
            cluster_0_count = sum((labels == 0) & (y == 1))
            cluster_1_count = sum((labels == 1) & (y == 1))
            if cluster_0_count > cluster_1_count:
                labels = 1 - labels

            # Calculate confusion matrix and accuracy metrics
            tn, fp, fn, tp = confusion_matrix(y, labels).ravel()
            accuracy = tp / (tp + fn)  # Accuracy for label 1
            
            # Calculate percentages
            cluster_0_percentage = sum(labels == 0) / len(labels)
            cluster_1_percentage = sum(labels == 1) / len(labels)

            # Identify top 20 samples closest to the cluster 1 center with original label 0
            cluster_1_center = X[(labels == 1) & (y == 0)].mean(axis=0)
            distances_cluster_1 = np.linalg.norm(X[(labels == 1) & (y == 0)] - cluster_1_center, axis=1)
            top_20_indices_cluster_1 = np.argsort(distances_cluster_1)[:50]
            top_20_ids_cluster_1 = ids[(labels == 1) & (y == 0)][top_20_indices_cluster_1]

            # Identify top 20 samples closest to the cluster 0 center
            cluster_0_center = X[labels == 0].mean(axis=0)
            distances_cluster_0 = np.linalg.norm(X[labels == 0] - cluster_0_center, axis=1)
            top_20_indices_cluster_0 = np.argsort(distances_cluster_0)[:50]
            top_20_ids_cluster_0 = ids[labels == 0][top_20_indices_cluster_0]

            # Identify samples in Cluster 1 (RNA-like) and Cluster 0 (Non-RNA-like)
            cluster_1_samples = ids[labels == 1]
            cluster_0_samples = ids[labels == 0]

            # Calculate clustering metrics
            silhouette_avg = silhouette_score(X, labels)
            ari = adjusted_rand_score(y, labels)
            ami = adjusted_mutual_info_score(y, labels)
            homogeneity = homogeneity_score(y, labels)
            completeness = completeness_score(y, labels)
            v_measure = v_measure_score(y, labels)

            metrics.append({
                "Method": name,
                "True Negatives": tn,
                "False Positives": fp,
                "False Negatives": fn,
                "True Positives": tp,
                "Accuracy": accuracy,
                "Cluster 0 Percentage": cluster_0_percentage,
                "Cluster 1 Percentage": cluster_1_percentage,
                "Silhouette Score": silhouette_avg,
                "Adjusted Rand Index": ari,
                "Adjusted Mutual Information": ami,
                "Homogeneity Score": homogeneity,
                "Completeness Score": completeness,
                "V-Measure Score": v_measure
            })

            top_20_cluster_1[name] = top_20_ids_cluster_1
            top_20_cluster_0[name] = top_20_ids_cluster_0
            cluster_1_samples_dict[name] = cluster_1_samples
            cluster_0_samples_dict[name] = cluster_0_samples

        except Exception as e:
            print(f"Error with {name}: {e}")

    return metrics, top_20_cluster_1, top_20_cluster_0, cluster_1_samples_dict, cluster_0_samples_dict

def save_results(results_dir, metrics, top_20_cluster_1, top_20_cluster_0, cluster_1_samples_dict, cluster_0_samples_dict):
    # Save metrics to a single CSV file
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(results_dir, 'clustering_metrics_se.csv'), index=False)

    # Save top 20 sample IDs in cluster 1 to a single CSV file
    top_20_cluster_1_df = pd.DataFrame.from_dict(top_20_cluster_1, orient='index').transpose()
    top_20_cluster_1_df.to_csv(os.path.join(results_dir, 'top_20_cluster_1_samples_se.csv'), index=False)

    # Save top 20 sample IDs in cluster 0 to a single CSV file
    top_20_cluster_0_df = pd.DataFrame.from_dict(top_20_cluster_0, orient='index').transpose()
    top_20_cluster_0_df.to_csv(os.path.join(results_dir, 'top_20_cluster_0_samples_se.csv'), index=False)

    # Save samples in Cluster 1 (RNA-like) for each clustering method to separate CSV files
    for method, samples in cluster_1_samples_dict.items():
        cluster_1_samples_df = pd.DataFrame(samples, columns=['IDs'])
        cluster_1_samples_df.to_csv(os.path.join(results_dir, f'cluster_1_samples_{method}_se.csv'), index=False)

    # Save samples in Cluster 0 (Non-RNA-like) for each clustering method to separate CSV files
    for method, samples in cluster_0_samples_dict.items():
        cluster_0_samples_df = pd.DataFrame(samples, columns=['IDs'])
        cluster_0_samples_df.to_csv(os.path.join(results_dir, f'cluster_0_samples_{method}_se.csv'), index=False)

def visualize_clustering_results(X_scaled, y, ids, clustering_methods, imgs_dir):
    for name, method in clustering_methods.items():
        try:
            if name == "GMM":
                labels = method.fit_predict(X_scaled)
            else:
                labels = method.fit(X_scaled).labels_

            # Assign clusters: cluster with most RNA existing structures (label 1) as cluster 1
            cluster_0_count = sum((labels == 0) & (y == 1))
            cluster_1_count = sum((labels == 1) & (y == 1))
            if cluster_0_count > cluster_1_count:
                labels = 1 - labels

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
            pca_df['labels'] = y.astype(str)
            pca_df['cluster'] = labels.astype(str)

            # Create a new column for legend labels
            pca_df['legend_labels'] = pca_df.apply(lambda row: f'Original label {row["labels"]}', axis=1)
            pca_df['legend_cluster'] = pca_df.apply(lambda row: f'Predicted as {row["cluster"]}', axis=1)

            # Custom color and symbol maps and set opacity  
            color_discrete_map = {
                'Predicted as 1': '#FD8984', # red
                'Predicted as 0': '#999999', # grey
            }

            symbol_map = {
                'Original label 0': 'x',
                'Original label 1': 'circle',
            }

            # Create scatter plot with Plotly
            fig = go.Figure()

            # Add 'x' markers for 'Original label 0' predicted as 0
            fig.add_trace(
                go.Scatter(
                    x=pca_df[(pca_df['legend_labels'] == 'Original label 0') & (pca_df['legend_cluster'] == 'Predicted as 0')]['PC1'],
                    y=pca_df[(pca_df['legend_labels'] == 'Original label 0') & (pca_df['legend_cluster'] == 'Predicted as 0')]['PC2'],
                    mode='markers',
                    marker=dict(symbol='x', color='#999999', size=8, opacity=0.2),
                    name='Original label 0 - Predicted as 0'
                )
            )

            # Add 'x' markers for 'Original label 0' predicted as 1
            fig.add_trace(
                go.Scatter(
                    x=pca_df[(pca_df['legend_labels'] == 'Original label 0') & (pca_df['legend_cluster'] == 'Predicted as 1')]['PC1'],
                    y=pca_df[(pca_df['legend_labels'] == 'Original label 0') & (pca_df['legend_cluster'] == 'Predicted as 1')]['PC2'],
                    mode='markers',
                    marker=dict(symbol='x', color='#FD8984', size=8, opacity=0.2),
                    name='Original label 0 - Predicted as 1'
                )
            )

            # Add 'circle' markers for 'Original label 1' predicted as 0
            fig.add_trace(
                go.Scatter(
                    x=pca_df[(pca_df['legend_labels'] == 'Original label 1') & (pca_df['legend_cluster'] == 'Predicted as 0')]['PC1'],
                    y=pca_df[(pca_df['legend_labels'] == 'Original label 1') & (pca_df['legend_cluster'] == 'Predicted as 0')]['PC2'],
                    mode='markers',
                    marker=dict(symbol='circle', color='#999999', size=10, opacity=0.8, line=dict(width=2, color='Black')),
                    name='Original label 1 - Predicted as 0'
                )
            )

            # Add 'circle' markers for 'Original label 1' predicted as 1
            fig.add_trace(
                go.Scatter(
                    x=pca_df[(pca_df['legend_labels'] == 'Original label 1') & (pca_df['legend_cluster'] == 'Predicted as 1')]['PC1'],
                    y=pca_df[(pca_df['legend_labels'] == 'Original label 1') & (pca_df['legend_cluster'] == 'Predicted as 1')]['PC2'],
                    mode='markers',
                    marker=dict(symbol='circle', color='#FD8984', size=10, opacity=0.8, line=dict(width=2, color='red')),
                    name='Original label 1 - Predicted as 1'
                )
            )

            # Update traces for different symbols with different opacities
            fig.update_traces(marker=dict(size=28, opacity=0.5), selector=dict(marker_symbol='x'))
            fig.update_traces(marker=dict(size=30, opacity=0.9, line=dict(width=3)), selector=dict(marker_symbol='circle'))
            fig.update_layout(showlegend=False)
            fig.update_layout(
                template='simple_white',
                title={'x': 0.5, 'xanchor': 'center'},
                # xaxis_title='Principal Component 1',
                # yaxis_title='Principal Component 2',
                xaxis_title='',
                yaxis_title='',
                font=dict(size=45),
                legend_title_text='',
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=0.9,
                    xanchor='right',
                    x=1
                ),
                width=800,  # Ensure the figure is square
                height=800
            )

            # Save the plot
            fig.write_image(os.path.join(imgs_dir, f'pca_plot_{name}.pdf'))
            # fig.show()
        except Exception as e:
            print(f"Error visualizing {name} method: {e}")

if __name__ == "__main__":
    vertices_num = int(sys.argv[1])  # 4,5,6,7,8,9, 0 for all
    data_collected = sys.argv[2]  # old, new

    # Load the data
    data_filtered, results_dir, imgs_dir = load_data(vertices_num, data_collected)

    # Preprocess the data
    X, y, ids = preprocess_data(data_filtered)

    # Define clustering methods
    clustering_methods = define_clustering_methods()

    # Apply clustering and evaluate
    metrics, top_20_cluster_1, top_20_cluster_0, cluster_1_samples_dict, cluster_0_samples_dict = evaluate_clustering(X, y, ids, clustering_methods)

    # Save results
    save_results(results_dir, metrics, top_20_cluster_1, top_20_cluster_0, cluster_1_samples_dict, cluster_0_samples_dict)

    # # Visualize clustering results
    # visualize_clustering_results(X, y, ids, clustering_methods, imgs_dir)
    print("Clustering evaluation completed, number of features used: ", X.shape[1])



## For testing
# python ml_algorithms.py 45 new