# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2024-11-04 14:41:15
LastModifiedBy: Rui Wang
LastEditTime: 2024-11-04 14:45:00
Email: wang.rui@nyu.edu
FilePath: /RNAClustering/src/seq2ct.py
Description: 
'''
import os
import sys


seq_id = sys.argv[1]


# Run NUPACK to Predict Structure:
os.system("mfe -pseudo -material rna" + seq_id + "2>/dev/null") 

# Extract the Folded Structure from Output
with open(seq_id+".mfe", 'r') as f:  
    fold = f.readlines()[16]

with open(seq_id+".mfe", 'w') as f:
    f.write(">seq\n")
    f.write(seq + "\n")
    f.write(fold)