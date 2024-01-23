from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB
import torch
import torch.nn as nn
import numpy as np
import os
import math
import time
import random
import torch.multiprocessing as multiprocessing

from opt_alg.rl_branch_and_CPT_alg import RLBranchAndCuttingPlaneTreeAlgorithm
from rl_alg.model import var_sorter,var_decider,SharedAdam,Critic

# instanceName = '/mnt/nfs_share/public/sribd_mip_library/2017-MIPLIB_collection/50v-10'
instanceName = '50v-10'


mdl1=var_sorter(v_size=6,c_size=3, sample_sizes=[64,128], multi_head=2, natt=2)
mdl1.share_memory()
mdl2=var_decider(v_size=6,c_size=3, sample_sizes=[64,128], multi_head=2, natt=2)
mdl2.share_memory()
mdl3=Critic(input_size=128*3, sample_sizes=[64,64])
mdl3.share_memory()
global_mdl=[mdl1,
            mdl2,
           mdl3, 
            SharedAdam(list(mdl1.parameters())+list(mdl2.parameters())+list(mdl3.parameters()))]



def train1(idx,lck):
    print(f'Adding worker {idx}...')
    bcpt = RLBranchAndCuttingPlaneTreeAlgorithm(instanceName, lock = lck ,maxIteration = 200, OutputFlag = 0, maxBuffer=5, Threads = 1, MIPGap = 0.0, TimeLimit = 300, nodeSelectionMode = 'BBR', branchVariableSelectionMode = 'MFR', cptVariableSelectionMode = 'MFR')
    bcpt.train(20)
    print('Done')

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    lck=multiprocessing.Lock()
    
    processes=[]
    n_workers=4

    for i in range(n_workers):
        p = multiprocessing.Process(target=train1,args=(i,lck))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # bcpt.solve(20)
