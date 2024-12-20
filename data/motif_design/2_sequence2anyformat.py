# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2024-11-09 01:19:14
LastModifiedBy: Rui Wang
LastEditTime: 2024-11-17 00:26:48
Email: wang.rui@nyu.edu
FilePath: /RNAClustering/data/motif_design/2_sequence2anyformat.py
Description: 
'''


import os
import sys

class SEQ2CTDOT:
    def __init__(self, seq_id, ct):
        self.seq_id = seq_id
        self.ct = ct

    def pknots(self):
        os.system(f"pknots -g {self.seq_id}.in {self.seq_id}_pknots.ct 2")
        with open(f'{self.seq_id}_pknots.ct','r+') as f:
            lines = f.readlines()
        with open(f'{self.seq_id}_pknots.ct','w') as f:
            f.write("seq\n")
            for i in range(4,len(lines)):
                f.write(lines[i])
        os.system(f"ct2dot {self.seq_id}_pknots.ct 1 {self.seq_id}_pknots.out")

    def ipknot(self):
        os.system(f"ipknot -g 2 -g 16 -e CONTRAfold -r 1 {self.seq_id}.in > {self.seq_id}_IPknot.txt")
        os.system(f"dot2ct {self.seq_id}_IPknot.txt {self.seq_id}_ipknot.ct")

    def nupack(self):
        os.system(f"mse -pseudo -material rna {self.seq_id} 2")
        os.system(f"dot2ct {self.seq_id}.mfe {self.seq_id}_nupack.ct")


# # In terminal 
# export NUPACKHOME=/workspace/nupack3.2.2

# # add NUPACKHOME to your path:
# export PATH=$PATH:$NUPACKHOME