# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2024-12-11 16:37:17
LastModifiedBy: Rui Wang
LastEditTime: 2024-12-11 16:53:08
Email: wang.rui@nyu.edu
FilePath: /RNAClustering/design/4_17.py
Description: 
'''

import re
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

# Reference sequence 
reference_sequence = "GGCGUCACACCUUCGGGUGAAGUCGCCCGAACUUCGCGCGAACUUCGCGCGAACUUCGCGCGAACUUCGCG" #4_17
file_path = './4_17_min_mut'


# Read and process the mutation sets from the file
with open(file_path, 'r') as file:
    mutation_lines = file.readlines()

# Create a FASTA record for the reference sequence
fasta_records = [SeqRecord(Seq(reference_sequence), id="reference", description="Reference sequence")]

# Extract mutations and create mutated sequences
for idx, line in enumerate(mutation_lines):
    # Remove leading/trailing whitespace and newlines
    line = line.strip()

    # Skip empty lines
    if not line:
        continue

    # Extract mutations using regular expression
    # Example format: "10C-U, 40G-C, ..."
    matches = re.findall(r'(\d+)([AUCG])\-([AUCG])', line)
    mutated_sequence = list(reference_sequence)  # Convert to list for mutability

    for match in matches:
        position = int(match[0]) - 1  # Convert 1-based index to 0-based
        original_base = match[1]  # Original base (not validated in this script)
        mutated_base = match[2]  # Mutated base
        # Ensure the original base matches the reference sequence for validation (optional)
        if mutated_sequence[position] != original_base:
            print(f"Warning: Position {position+1} expected {original_base}, found {mutated_sequence[position]} in reference.")
        mutated_sequence[position] = mutated_base  # Apply mutation

    # Create a FASTA record
    mutated_seq_str = "".join(mutated_sequence)
    record = SeqRecord(Seq(mutated_seq_str), id=f"mutant_{idx+1}", description=f"Mutation set {idx+1}")
    fasta_records.append(record)

# Write to a FASTA file
output_file = "./4_17_mutated_sequences.fasta"
SeqIO.write(fasta_records, output_file, "fasta")


from dash import Dash, html
import dash_bio as dashbio
from selenium import webdriver
import time

app = Dash(__name__)

# Load alignment data
import urllib.request as urlreq
# data = urlreq.urlopen('https://git.io/alignment_viewer_p53.fasta').read().decode('utf-8')
# print(data)
with open('./4_17_mutated_sequences.fasta', 'r') as file:
    data = file.read()

# Dash layout
app.layout = html.Div([
    dashbio.AlignmentChart(
        id='alignment-viewer',
        data=data
    ),
])

if __name__ == '__main__':
    app.run_server(debug=True)
