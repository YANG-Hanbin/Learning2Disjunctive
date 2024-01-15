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

from opt_alg.branch_and_bound_framework import BranchAndBoundFramework


"""
    The class BranchAndDisjunctiveCutAlgorithm is a subclass of BranchAndBoundFramework, which implements the branch-and-CPT algorithm.
    - It allows to select multiple branching variables instead of one branching variable in each iteration.
"""
class BranchAndCuttingPlaneTreeAlgorithm(BranchAndBoundFramework):
    def __init__(self, *args, number_branch_var = 2, number_branch_node = 1, **kwargs):
        # Initialize parent class with all arguments passed
        super().__init__(*args, **kwargs)
        # Initialization for new attributes
        self.number_branch_var = number_branch_var
        self.number_branch_node = number_branch_node


    def branch_variable_selection(self):
        """
            Select a set of variables to branch

            return: self.branchVar, a dictionary with key: variable name, value: variable value
        """
        # super().branch_variable_selection()  # method overriding: You can call the original method, too, if needed
        number_of_candidates = self.number_branch_var  # the number of variables that are chosen to branch, so the number of nodes in the branching tree is 2^number_of_candidates
        node = self.branch_bound_tree[self.branch_node]
        number_of_noninteger = len(node['fractional_int'])
        number_of_nonbinary = len(node['fractional_bin'])

        self.branchVar = {}
        if number_of_noninteger > 0:
            list1 = sorted(node['fractional_int'].items(), key=lambda x: x[1], reverse=True)[:number_of_candidates]  # find the integer variables that have the largest distance to the nearest integer
            if len(list1) <= number_of_candidates:
                for item in list1:
                    self.branchVar[item[0]] = item[1]  #
            else:
                for item in list1[0:number_of_candidates]:
                    self.branchVar[item[0]] = item[1]

        if number_of_nonbinary > 0:
            list2 = sorted(node['fractional_bin'].items(), key=lambda x: x[1], reverse=True)[:number_of_candidates]  # find the binary variables that have the largest distance to {0,1}
            if len(list2) <= number_of_candidates:
                for item in list2:
                    self.branchVar[item[0]] = item[1]
            else:
                for item in list2[0:number_of_candidates]:
                    self.branchVar[item[0]] = item[1]
        

    def branching_tree_building(self, level, branch_node, sol):
        """
            Auxiliary function for the method self.branching()
            
            Build the branching tree
        """
        # create two new nodes with info in self.branch_node, del self.branch_node, update two new nodes with the
        # method self.nodal_problem
        if level == len(self.branchVar.keys()):
            self.nodeSet[branch_node] = None
            self.nodal_problem(branch_node)
            return
        else:
            branch_variable = list(self.branchVar.keys())[level]
            node = self.branch_bound_tree[branch_node]  # father node
            pos = self.varName_map_position[branch_variable]
            left_node = {}
            left_node['cuts'] = deepcopy(node['cuts'])
            left_node['cutA'] = deepcopy(node['cutA'])
            left_node['cutRHS'] = deepcopy(node['cutRHS'])
            left_node['trace'] = deepcopy(node['trace'])
            left_node['trace'].append([branch_variable, '<', math.floor(sol[pos])])  # x <= floor(xhat)

            right_node = {}
            right_node['cuts'] = deepcopy(node['cuts'])
            right_node['cutA'] = deepcopy(node['cutA'])
            right_node['cutRHS'] = deepcopy(node['cutRHS'])
            right_node['trace'] = deepcopy(node['trace'])
            right_node['trace'].append([branch_variable, '>', math.ceil(sol[pos])])  # x >= ceil(xhat)

            left_node_ind = max(self.branch_bound_tree.keys()) + 1
            right_node_ind = left_node_ind + 1
            self.branch_bound_tree[left_node_ind] = left_node
            self.branch_bound_tree[right_node_ind] = right_node
            del self.branch_bound_tree[branch_node]
            self.branching_tree_building(level + 1, left_node_ind, sol)
            self.branching_tree_building(level + 1, right_node_ind, sol)

    def branching(self):
        """
            Build the branching tree by naive CPT algorithm
            In other words, this is "Parent-Based Branching".

            return: self.nodeSet, a dictionary with key: node_index, value: node info
        """
        self.nodeSet = {}
        self.branching_tree_building(0, self.branch_node, self.branch_bound_tree[self.branch_node]['sol'])
    
    def cut_generation_node_selection(self):
        """
            Select a set of nodes to generate cuts
            In other words, this is "nodal selection".

            return: self.nodeSet, a dictionary with key: node_index, value: node info
        """
        # TODO:: Using RL to selection a set of nodes to generate cuts
        nodeSet = None # the set of node_index to generate cuts
        if self.cglpNodeSelectionModel == 'bound-based':                                            
            node_values = [(node_index, node['value']) for node_index, node in self.branch_bound_tree.items()]
            node_values.sort(key=lambda x: x[1])
            nodeSet = [node_index for node_index, value in node_values[:self.nodeNumber]]
            # nodeSet = self.branch_bound_tree.keys() 
        elif self.cglpNodeSelectionModel == 'parentnode-based':                                     
            nodeSet = list(self.nodeSet.keys())
        elif self.cglpNodeSelectionModel == 'RL-based':
            
            pass

        self.nodeSet = {} # initialize the set of node with info to generate cuts
        # add bounds to nodes in nodeSet
        if nodeSet != None:
            for node_index in nodeSet:
                    # add bounds to this node
                    node = {}
                    tmp_LB = deepcopy(self.LB)
                    tmp_UB = deepcopy(self.UB)
                    for item in self.branch_bound_tree[node_index]['trace']:
                        varName, sense, bound = item
                        pos = self.varName_map_position[varName]
                        if sense == '<':
                            tmp_UB[pos] = min(tmp_UB[pos], bound)
                        elif sense == '>':
                            tmp_LB[pos] = max(tmp_LB[pos], bound)
                    node['LB'] = tmp_LB
                    node['UB'] = tmp_UB
                    self.nodeSet[node_index] = node
        else:
            print(f'nodeSet is None')
            exit()


    def solve(self):
        time_init = time.time()
        while self.iteration <= self.maxIteration:
            iter_begin = time.time()
            # selection a node to branch
            if self.iteration == 0:
                self.branch_node_selection()
            if self.OPT == True or self.upper_bound['value'] - self.lower_bound['value'] <= 1e-2:
                self.print_iteration_info()
                return
            # selection a variable to branch
            self.branch_variable_selection()
            # branch
            self.branching()
            # bound
            self.fathom_by_bounding()
            # multiple branching if needed
            for i in range(self.number_branch_node - 1):
                self.branch_node_selection()
                self.branch_variable_selection()
                self.branching()
                self.fathom_by_bounding()

            # select nodes to generate cuts
            self.cut_generation_node_selection()
            ready_to_cut = time.time()
            # generate cuts
            self.cut_generation()
            iter_end = time.time()
            overall = iter_end - time_init
            iteration_time = iter_end - iter_begin
            cut_time = iter_end - ready_to_cut
            self.branch_node_selection()
            self.iteration += 1
            self.print_iteration_info(cut_time, iteration_time, overall)
