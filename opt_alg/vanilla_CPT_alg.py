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

from opt_alg.cutting_plane_framework import CuttingPlaneMethod
from rl_alg.model import var_sorter

class CuttingPlaneTreeAlgorithm(CuttingPlaneMethod):
    def __init__(self, instanceName, maxIteration=100, OutputFlag=0, Threads=1, MIPGap=0.0, TimeLimit=3600, MIPFocus=2, cglp_OutputFlag=0, cglp_Threads=1, cglp_MIPGap=0.0, cglp_TimeLimit=100, cglp_MIPFocus=0, addCutToMIP=False, number_branch_var=2, normalization='SNC', additional_param=None):
        super().__init__(instanceName, maxIteration, OutputFlag, Threads, MIPGap, TimeLimit, MIPFocus, cglp_OutputFlag, cglp_Threads, cglp_MIPGap, cglp_TimeLimit, cglp_MIPFocus, addCutToMIP, number_branch_var, normalization)
        self.additional_param = additional_param
    
    def locate_node(self):
        for node_index, node in self.nodeSet.items():
            if node['branchInfo'] == []:
                return node_index
            for info in node['branchInfo']: # info <- (varName, SENSE, bound)
                pos = self.varName_map_position[info[0]]
                if info[1] == '<':
                    if self.lp_sol[pos] <= info[2]:
                        continue
                    else:
                        break
                elif info[1] == '>':
                    if self.lp_sol[pos] >= info[2]:
                        print('2')
                        continue
                    else:
                        break
                return node_index
        
        return random.choice(list(self.nodeSet.keys()))
    

    def branching_variable_selection(self):
        # TODO::add ML model to choose the variable to branch
        # choose the variable to branch: Maximum Fractionality Rule
        number_of_noninteger = len(self.non_integer_vars[self.iteration - 1])
        number_of_nonbinary = len(self.non_binary_vars[self.iteration - 1])

        self.branchVar[self.iteration-1] = {}
        maxKey = None
        maxDistance = None
        if number_of_noninteger > 0:  
            maxKey = max(self.non_integer_vars[self.iteration-1], key=self.non_integer_vars[self.iteration-1].get)     # find the integer variables that have the largest distance to the nearest integer  
            maxDistance = self.non_integer_vars[self.iteration-1][maxKey] 

        if number_of_nonbinary > 0: 
            tmp_maxKey = max(self.non_binary_vars[self.iteration-1], key=self.non_binary_vars[self.iteration-1].get)     # find the integer variables that have the largest distance to the nearest integer  
            tmp_maxDistance = self.non_binary_vars[self.iteration-1][tmp_maxKey] 
            if maxDistance == None or tmp_maxDistance > maxDistance:
                maxKey = tmp_maxKey
                maxDistance = tmp_maxDistance

        self.branchVar[self.iteration-1][maxKey] = maxDistance
        return maxKey


    def cutting_plane_tree(self):     
        # 0. initialize the cutting plane tree
        if self.iteration == 1:
            self.nodeSet = {}
            self.nodeSet[0] = {}
            self.nodeSet[0]['LB'] = deepcopy(self.LB)
            self.nodeSet[0]['UB'] = deepcopy(self.UB)
            self.nodeSet[0]['branchInfo'] = [] # node[branchInfo] <- (varName, SENSE, bound)
        # 1. locate the LP solution in the cutting plane tree
        node = self.locate_node()
        # print('node: ', node)
        if node == None:
            return 
        # 2. select the variables to branch
        varName = self.branching_variable_selection()
        pos = self.varName_map_position[varName]

        # 3. update the cutting plane tree
        left_node = {}
        left_node['LB'] = deepcopy(self.nodeSet[node]['LB'])
        left_node['LB'][pos] = math.ceil(self.lp_sol[pos])  
        left_node['UB'] = deepcopy(self.nodeSet[node]['UB'])
        left_node['branchInfo'] = deepcopy(self.nodeSet[node]['branchInfo'])
        left_node['branchInfo'].append([varName, '>', math.ceil(self.lp_sol[pos])])

        right_node = {}
        right_node['LB'] = deepcopy(self.nodeSet[node]['LB'])
        right_node['UB'] = deepcopy(self.nodeSet[node]['UB'])
        right_node['UB'][pos] = math.floor(self.lp_sol[pos]) 
        right_node['branchInfo'] = deepcopy(self.nodeSet[node]['branchInfo'])
        right_node['branchInfo'].append([varName, '<', math.floor(self.lp_sol[pos])])
        # update the cutting plane tree by deleting the current node and adding two new children nodes
        left_node_ind = max(self.nodeSet.keys()) + 1
        right_node_ind = left_node_ind + 1
        self.nodeSet[left_node_ind] = left_node
        self.nodeSet[right_node_ind] = right_node
        del self.nodeSet[node]  


    def solve(self):
        time_init = time.time()
        while self.iteration <= self.maxIteration:
            iter_begin = time.time()
            self.master_problem()
            if self.OPT == True:
                self.print_iteration_info()
                return
            self.cutting_plane_tree()
            ready_to_cut = time.time()
            self.cut_generation()
            iter_end = time.time()
            overall = iter_end - time_init
            iteration_time = iter_end - iter_begin
            cut_time = iter_end - ready_to_cut
            self.print_iteration_info(cut_time, iteration_time, overall)