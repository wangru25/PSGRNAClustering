# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2024-06-24 13:07:34
LastModifiedBy: Rui Wang
LastEditTime: 2024-07-07 22:20:53
Email: wang.rui@nyu.edu
FilePath: /RNAClustering/src/text.py
Description: 
'''
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import os, sys
pio.templates.default = "simple_white"

vertices_num = int(sys.argv[1])
data_collected = sys.argv[2]  # old, new

if vertices_num == 0: 
    results_dir = f"../results/{data_collected}"
    imgs_dir = f"../img/clustering/{data_collected}"
else: 
    results_dir = f"../results/{data_collected}/{vertices_num}_vertices"
    imgs_dir = f"../img/clustering/{data_collected}/{vertices_num}_vertices"

# Load the clustering metrics data
# metrics_df = pd.read_csv('../results/clustering_metrics.csv')
metrics_df = pd.read_csv(os.path.join(results_dir, 'clustering_metrics.csv'))
print(metrics_df)
print(results_dir)

# Select the metrics to visualize
metrics_to_visualize = ['Method', 'Silhouette Score', 'Homogeneity Score']

# Separate Accuracy for an individual plot
accuracy_df = metrics_df[['Method', 'Accuracy']]

# Filter the dataframe to include only the selected metrics except Accuracy
other_metrics_df = metrics_df[metrics_to_visualize]

# Melt the dataframe for easier plotting with Plotly
melted_other_df = other_metrics_df.melt(id_vars=['Method'], var_name='Metric', value_name='Score')

# Create a bar plot for the other selected metrics
fig_other = px.bar(melted_other_df, x='Method', y='Score', color='Metric', barmode='group', 
                title='Clustering Method Performance Metrics (Excluding Accuracy)',
                labels={'Score': 'Metric Score', 'Method': 'Clustering Method'})

# Update layout for better readability
fig_other.update_layout(
    xaxis_title='Clustering Method',
    yaxis_title='Score',
    legend_title='Metrics',
    title_x=0.5,
    template='simple_white',
    width=1000,
    height=600
)

# Create a bar plot for Accuracy
fig_accuracy = px.bar(accuracy_df, x='Accuracy', y='Method', 
                    title='Clustering Method Accuracy',
                    orientation= 'h',
                    labels={'Accuracy': 'Accuracy Score', 'Method': 'Clustering Method'})

# Update layout for better readability
fig_accuracy.update_layout(
    xaxis_title='Clustering Method',
    yaxis_title='Accuracy',
    title_x=0.5,
    template='simple_white',
    width=800,
    height=600
)

# # Show the plots
# fig_other.show()
# fig_accuracy.show()
pio.write_image(fig_other, f'{imgs_dir}/clustering_metrics.pdf', width=1000, height=600)
pio.write_image(fig_accuracy, f'{imgs_dir}/clustering_accuracy.pdf', width=800, height=600)

# # Define a function to create a confusion matrix dataframe
# def create_confusion_matrix_df(metrics_df):
#     confusion_matrices = []
#     methods = metrics_df['Method']
#     for i, method in enumerate(methods):
#         tn = metrics_df.loc[i, 'True Negatives']
#         fp = metrics_df.loc[i, 'False Positives']
#         fn = metrics_df.loc[i, 'False Negatives']
#         tp = metrics_df.loc[i, 'True Positives']
#         confusion_matrix = pd.DataFrame({
#             'Predicted Negative': [tn, fn],
#             'Predicted Positive': [fp, tp]
#         }, index=['Actual Negative', 'Actual Positive'])
#         confusion_matrices.append((method, confusion_matrix))
#     return confusion_matrices

# # Create confusion matrices for each method
# confusion_matrices = create_confusion_matrix_df(metrics_df)

# # Define a custom colorscale based on specific ranges
# def custom_colorscale(value):
#     if value <= 10:
#         return 'rgb(255, 255, 255)'  # white
#     elif 10 < value <= 300:
#         return 'rgb(100, 149, 237)'  # cornflower blue
#     else:
#         return 'rgb(0, 0, 255)'  # blue

# # Plot confusion matrices
# for method, cm in confusion_matrices:
#     # Apply custom colorscale to the confusion matrix values
#     z_values = cm.values
#     custom_z_values = [[custom_colorscale(val) for val in row] for row in z_values]
    
#     fig = go.Figure(data=go.Heatmap(
#         z=z_values,
#         x=cm.columns,
#         y=cm.index,
#         colorscale = [
#             [0, 'rgb(247, 251, 255)'],
#             [0.0001, 'rgb(198, 219, 239)'],
#             [0.4, 'rgb(158, 202, 225)'],
#             [0.4444444444444444, 'rgb(107, 174, 214)'],
#             [0.5, 'rgb(33, 113, 181)'],
#             [0.7, 'rgb(8, 81, 156)'],
#             [1.0, 'rgb(8, 48, 107)']
#         ],

#         # colorscale=[[0, 'rgb(255, 255, 255)'], [0.1, 'rgb(255, 255, 255)'], [0.1, 'rgb(100, 149, 237)'], [0.9, 'rgb(100, 149, 237)'], [0.9, 'rgb(0, 0, 255)'], [1, 'rgb(0, 0, 255)']],
#         colorbar=dict(title='Count')
#     ))

#     fig.update_layout(
#         title=f'Confusion Matrix for {method}',
#         xaxis_title='Predicted Labels',
#         yaxis_title='Actual Labels',
#         width=600,
#         height=600
#     )
#     # fig.show()
#     # pio.write_image(fig, f'{imgs_dir}/confusion_matrix_{method}.pdf', width=600, height=600)





# Example run command:
# python src/text.py 45 new