# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2024-10-14 12:29:10
LastModifiedBy: Rui Wang
LastEditTime: 2024-11-09 05:08:32
Email: wang.rui@nyu.edu
FilePath: /RNAClustering/matchgraph/dualGraphCheck.py
Description: 
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 22:31:46 2020

@author: qiyaozhu
"""
import random as RANDOM
import os
import os.path
import sys
import time
from functools import partial
import multiprocessing
from ClassesFunctions import *
from dualGraphs import *


def graphFinder(jobID):
    
    RNA = getCTInfo(jobID+".ct")
    # os.system("rm -rf "+jobID+".ct")
    
    countHelices(RNA) 
    changeHelices(RNA)
    RNA.makeMatrices()
    connectHelices(RNA)
    for i in range(0,len(RNA.adjMatrix)): # S.J. 07/11/2018 - to keep track of vertexOrder
        vertexOrder.append(0)
        
    success, graph = calcEigen(RNA)
    correctHNumbers(RNA)
    if len(RNA.adjMatrix)==1 or len(RNA.adjMatrix)>9:
        print ("No matching graph exists because vertex number is either 1 or greater than 10.")
        return None
    elif success == 0: # no graph ID was assigned as eigen values not in the library S.J. 11/09/2017
        print ("No matching graph exists (even if the vertex number is between 2 and 9).")
        return None
    else:
        return graph
    
if __name__ == "__main__":    
    jobID = sys.argv[1]
    graph = graphFinder(jobID)
    
    if graph != None:
        print(graph)
        with open("graph"+jobID+".txt", "w") as f:
            f.write(graph)
    else:
        print("No matching graph exists.")
        with open("graph"+jobID+".txt", "w") as f:
            f.write("No matching graph exists.")
    sys.exit(0)