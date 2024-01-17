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
    def __init__(self, *args, number_branch_var=2, number_branch_node=1, cptVariableSelectionMode = 'MFR', **kwargs):
        # Initialize parent class with all arguments passed
        super().__init__(*args, **kwargs)
        # Initialization for new attributes
        self.number_branch_var = number_branch_var
        self.number_branch_node = number_branch_node
        self.cutting_plane_tree = {}                         # key: node_index <- value: (trace, cuts), all leaf nodes
        self.nodeSet = {}                                    # nodeSet[node] <- (LB, UB)
        self.cptVariableSelectionMode = cptVariableSelectionMode  # 'MFR', 'RAND' in CPT


    def naive_cpt_variable_selection(self):
        """
            Select a set of variables to branch in naive CPT algorithm

            return: self.branchVar, a dictionary with key: variable name, value: variable value
        """
        # super().branch_variable_selection()  # method overriding: You can call the original method, too, if needed
        number_of_candidates = self.number_branch_var  # the number of variables that are chosen to branch, so the number of nodes in the branching tree is 2^number_of_candidates
        node = self.branch_bound_tree[self.branch_node]
        number_of_noninteger = len(node['fractional_int'])
        number_of_nonbinary = len(node['fractional_bin'])
        if self.cptVariableSelectionMode == 'MFR':
            self.branchVar = {}
            if number_of_noninteger > 0:
                list1 = sorted(node['fractional_int'].items(), key=lambda x: x[1], reverse=True)[
                        :number_of_candidates]  # find the integer variables that have the largest distance to the nearest integer
                if len(list1) <= number_of_candidates:
                    for item in list1:
                        self.branchVar[item[0]] = item[1]  #
                else:
                    for item in list1[0:number_of_candidates]:
                        self.branchVar[item[0]] = item[1]

            if number_of_nonbinary > 0:
                list2 = sorted(node['fractional_bin'].items(), key=lambda x: x[1], reverse=True)[
                        :number_of_candidates]  # find the binary variables that have the largest distance to {0,1}
                if len(list2) <= number_of_candidates:
                    for item in list2:
                        self.branchVar[item[0]] = item[1]
                else:
                    for item in list2[0:number_of_candidates]:
                        self.branchVar[item[0]] = item[1]

        elif self.cptVariableSelectionMode == 'RAND':
            if number_of_noninteger > 0:
                list1 = list(node['fractional_int'].keys())
                self.branchVar = {random.choice(list1): number_of_candidates}
            elif number_of_nonbinary > 0:
                list2 = list(node['fractional_bin'].keys())
                self.branchVar = {random.choice(list2): number_of_candidates}
            else:
                print(f'node {self.branch_node} has no fractional variables')
                exit()


    def feasibilityTest(self, node_index, tree):  # node_index is a new child node
        # Create the nodal bounding model
        self.bounding_problem = self.lp_relaxation.copy()
        node = tree[node_index]
        cutSet = node['cuts']  # the set of cuts inherited from its parent's node
        trace = node['trace']  # the set of additional bound constraint along the branch-and-bound tree
        # add bounds to subproblem
        for item in trace:
            varName, sense, bound = item
            if sense == '<':
                self.bounding_problem.getVarByName(varName).ub = bound  # x <= floor(xhat) = ub
            elif sense == '>':
                self.bounding_problem.getVarByName(varName).lb = bound  # x >= ceil(xhat) = lb
            elif sense == '=':  # this is useless
                self.bounding_problem.getVarByName(varName).ub = bound
                self.bounding_problem.getVarByName(varName).lb = bound
        # add the previous cuts to subproblem
        if node['cutRHS'] != None:
            self.bounding_problem.addMConstr(node['cutA'], self.variables, GRB.GREATER_EQUAL, node['cutRHS'])
        # add the new cut to subproblem
        for i, cut in cutSet.items():
            piBest = cut['piBest']  # piBest is a dictionary
            pi0Best = cut['pi0Best']  # pi0Best is a float
            newCut = gp.LinExpr(- pi0Best)
            for var in self.varName:
                newCut += self.bounding_problem.getVarByName(var) * piBest[var]
            self.bounding_problem.addConstr(newCut >= 0.0)

        self.bounding_problem.update()
        
        self.bounding_problem.setObjective(0,GRB.MINIMIZE)
        self.bounding_problem.optimize()
        if self.bounding_problem.status == GRB.INFEASIBLE:
            # fathom infeasible nodes
            del tree[node_index]
            # print(f'fathom node {node_index} by infeasibility')
            return False
        return True


    def branching_and_cut_generation(self):
        """
            Branching on the variable self.branch_variable in the node self.branch_node.
        """
        # create two new nodes with info in self.branch_node, del self.branch_node, update two new nodes with the
        # method self.nodal_problem
        node = self.branch_bound_tree[self.branch_node]  # father node
        pos = self.varName_map_position[self.branch_variable]

        left_node = {}
        left_node['cuts'] = deepcopy(node['cuts'])
        left_node['trace'] = deepcopy(node['trace'])
        left_node['trace'].append([self.branch_variable, '<', math.floor(node['sol'][pos])])  # x <= floor(xhat)
        left_node['cutA'] = deepcopy(node['cutA'])
        left_node['cutRHS'] = deepcopy(node['cutRHS'])

        right_node = {}
        right_node['cuts'] = deepcopy(node['cuts'])
        right_node['trace'] = deepcopy(node['trace'])
        right_node['trace'].append([self.branch_variable, '>', math.floor(node['sol'][pos])+1])  # x >= ceil(xhat)
        right_node['cutA'] = deepcopy(node['cutA'])
        right_node['cutRHS'] = deepcopy(node['cutRHS'])

        left_node_ind = max(self.branch_bound_tree.keys()) + 1
        right_node_ind = left_node_ind + 1
        self.branch_bound_tree[left_node_ind] = left_node
        self.branch_bound_tree[right_node_ind] = right_node
        feasibilityLeft = self.feasibilityTest(left_node_ind, self.branch_bound_tree)
        feasibilityRight = self.feasibilityTest(right_node_ind, self.branch_bound_tree)
        
        # build a naive CPT tree for the left node (Node-Based Branching)
        if feasibilityLeft:
            self.cutting_plane_tree = {}
            self.cutting_plane_tree[left_node_ind] = left_node
            self.cutting_plane_tree_building(0, left_node_ind, self.branch_bound_tree[self.branch_node]['sol'])
            # select nodes to generate cuts
            self.cut_generation_node_selection()
            # generate cuts
            self.cut_generation(left_node_ind)
        # build a naive CPT tree for the right node (Node-Based Branching)
        if feasibilityRight:
            self.cutting_plane_tree = {}
            self.cutting_plane_tree[right_node_ind] = right_node
            self.cutting_plane_tree_building(0, right_node_ind, self.branch_bound_tree[self.branch_node]['sol'])
            # select nodes to generate cuts
            self.cut_generation_node_selection()
            # generate cuts
            self.cut_generation(right_node_ind)

        del self.branch_bound_tree[self.branch_node]

    def cutting_plane_tree_building(self, level, branch_node_ind, sol):
        """
            Auxiliary function for the method self.branching()
            
            Build the branching tree
        """
        # create two new nodes with info in self.branch_node, del self.branch_node, update two new nodes with the
        # method self.nodal_problem
        if level == len(self.branchVar.keys()):
            return
        else:
            branch_variable = list(self.branchVar.keys())[level]
            node = self.cutting_plane_tree[branch_node_ind]          # father node
            pos = self.varName_map_position[branch_variable]
            left_node = {}
            left_node['cuts'] = deepcopy(node['cuts'])
            left_node['trace'] = deepcopy(node['trace'])
            left_node['cutA'] = deepcopy(node['cutA'])
            left_node['cutRHS'] = deepcopy(node['cutRHS'])
            left_node['trace'].append([branch_variable, '<', math.floor(sol[pos])])  # x <= floor(xhat)

            right_node = {}
            right_node['cuts'] = deepcopy(node['cuts'])
            right_node['trace'] = deepcopy(node['trace'])
            right_node['cutA'] = deepcopy(node['cutA'])
            right_node['cutRHS'] = deepcopy(node['cutRHS'])
            right_node['trace'].append([branch_variable, '>', math.floor(sol[pos])+1])  # x >= ceil(xhat)

            left_node_ind = max(self.cutting_plane_tree.keys()) + 1
            right_node_ind = left_node_ind + 1
            self.cutting_plane_tree[left_node_ind] = left_node
            self.cutting_plane_tree[right_node_ind] = right_node

            feasibilityLeft = self.feasibilityTest(left_node_ind, self.cutting_plane_tree)
            feasibilityRight = self.feasibilityTest(right_node_ind, self.cutting_plane_tree)
            if feasibilityLeft:
                self.cutting_plane_tree_building(level + 1, left_node_ind, sol)
            if feasibilityRight:
                self.cutting_plane_tree_building(level + 1, right_node_ind, sol)
            del self.cutting_plane_tree[branch_node_ind]


    def cut_generation_node_selection(self):
        """
            Select a set of nodes to generate cuts
            There are three rules to select nodes to generate cuts:
            - Bound-based Rule: select the nodes with the smallest lower bounds
            - Parent Node-based Rule: select the all child nodes of the branching node (self.branch_node)
            - RL-based Rule: use RL to select a set of nodes to generate cuts

            return self.nodeSet
        """
        # TODO:: Using RL to selection a set of nodes to generate cuts
        # here we use up to 3 most promising nodes to generate cut
        self.nodeSet = {}
        nodeSet = list(self.cutting_plane_tree.keys())

        # add bounds to nodes in nodeSet
        if nodeSet != None:
            for node_index in nodeSet:
                    # add bounds to this node
                    node = {}
                    tmp_LB = deepcopy(self.LB)
                    tmp_UB = deepcopy(self.UB)
                    for item in self.cutting_plane_tree[node_index]['trace']:
                        varName, sense, bound = item
                        pos = self.varName_map_position[varName]
                        if sense == '<':
                            tmp_UB[pos] = min(tmp_UB[pos], bound)
                        elif sense == '>':
                            tmp_LB[pos] = max(tmp_LB[pos], bound)
                    node['LB'] = tmp_LB
                    node['UB'] = tmp_UB
                    # node['cutA'] = self.branch_bound_tree[node_index]['cutA']
                    # node['cutRHS'] = self.branch_bound_tree[node_index]['cutRHS']
                    self.nodeSet[node_index] = node
        else:
            print(f'nodeSet is None')
            exit()


    def cut_generation(self, child_node_index):
        self.nodal_problem(child_node_index)
        cglp = gp.Model("cglp")
        # Create variables
        pi = cglp.addVars(self.varName, vtype=GRB.CONTINUOUS, lb=-float('inf'), name=f"pi", obj=0.0)
        pi0 = cglp.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), name=f'pi_0', obj=-1.0)
        cglp.update()
        # Set objective
        sol = self.branch_bound_tree[child_node_index]['sol']
        # sol = self.lower_bound_sol
        cglp.setObjective(
                gp.quicksum(pi[v] * sol[self.varName_map_position[v]] for v in self.varName) - pi0,
                GRB.MINIMIZE)
        if self.incumbent != None:
            self.nodeSelectionMode = 'BBR'

        num_constrs, num_vars = self.A.shape[0], self.A.shape[1]  # self.mipModel.getAttr('NumConstrs'), self.mipModel.getAttr('NumVars')
        cglp_lambda = {}
        cglp_mu = {}
        cglp_v = {}
        cglp_gamma = {}
        # cglp normalization constraint
        normalizationConstraint = gp.LinExpr(-1.0)
        for node_index, node in self.nodeSet.items():
            cglp_lambda[node_index] = []
            cglp_mu[node_index] = []
            cglp_v[node_index] = []
            cglp_gamma[node_index] = [] 

            num_cuts = len(self.cutting_plane_tree[node_index]['cutRHS']) if self.cutting_plane_tree[node_index]['cutRHS'] != None else 0
            # cglp equation (3) in ORL paper
            ineqConstraint = gp.LinExpr(-pi0)
            for i in range(num_constrs):
                sense = self.SENSE[i]
                if sense == '<':
                    cglp_lambda[node_index].append(
                        cglp.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f'lambda_{node_index}_{i}', obj=0.0))
                    ineqConstraint.addTerms(-self.RHS[i], cglp_lambda[node_index][i])
                    normalizationConstraint.addTerms(1, cglp_lambda[node_index][i])
                elif sense == '>':
                    cglp_lambda[node_index].append(
                        cglp.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f'lambda_{node_index}_{i}', obj=0.0))
                    ineqConstraint.addTerms(self.RHS[i], cglp_lambda[node_index][i])
                    normalizationConstraint.addTerms(1, cglp_lambda[node_index][i])
                elif sense == '=':
                    # Ax = b <=> Ax >= b and -Ax >= -b
                    cglp_lambda[node_index].append(
                        cglp.addVars(['+', '-'], vtype=GRB.CONTINUOUS, lb=0.0, name=f'lambda_{node_index}_{i}',
                                     obj=0.0))
                    ineqConstraint.addTerms(self.RHS[i], cglp_lambda[node_index][i]['+'])
                    normalizationConstraint.addTerms(1, cglp_lambda[node_index][i]['+'])

                    ineqConstraint.addTerms(-self.RHS[i], cglp_lambda[node_index][i]['-'])
                    normalizationConstraint.addTerms(1, cglp_lambda[node_index][i]['-'])

            for i in range(num_vars):
                var = self.varName[i]
                cglp_mu[node_index].append(
                    cglp.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f'mu_{node_index}_{var}', obj=0.0))
                cglp_v[node_index].append(
                    cglp.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f'v_{node_index}_{var}', obj=0.0))
                ineqConstraint.addTerms(node['LB'][i], cglp_mu[node_index][i])
                ineqConstraint.addTerms(-node['UB'][i], cglp_v[node_index][i])
                normalizationConstraint.addTerms(1, cglp_mu[node_index][i])
                normalizationConstraint.addTerms(1, cglp_v[node_index][i])

            for i in range(num_cuts):
                cglp_gamma[node_index].append(
                        cglp.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f'gamma_{node_index}_{i}', obj=0.0))
                ineqConstraint.addTerms(self.cutting_plane_tree[node_index]['cutRHS'][i], cglp_gamma[node_index][i])
                normalizationConstraint.addTerms(1, cglp_gamma[node_index][i])

            cglp.addConstr(ineqConstraint >= 0, name=f'equation3_{node_index}')

            # cglp equation (2) in ORL paper
            for i in range(num_vars):
                var = self.varName[i]
                eqConstraint = gp.LinExpr(-pi[var])
                eqConstraint.addTerms(1, cglp_mu[node_index][i])
                eqConstraint.addTerms(-1, cglp_v[node_index][i])

                ## add constraint matrix multiplication term
                constr_index = self.A.getcol(i).nonzero()[0]  # the set of constraints that contain the variable 'var'
                for j in constr_index:
                    sense = self.SENSE[j]
                    if sense == '<':
                        eqConstraint.addTerms(-self.A[j, i], cglp_lambda[node_index][j])
                    elif sense == '>':
                        eqConstraint.addTerms(self.A[j, i], cglp_lambda[node_index][j])
                    elif sense == '=':
                        eqConstraint.addTerms(self.A[j, i], cglp_lambda[node_index][j]['+'])
                        eqConstraint.addTerms(-self.A[j, i], cglp_lambda[node_index][j]['-'])

                ## add cut matrix multiplication term
                constr_index = self.cutting_plane_tree[node_index]['cutA'].getcol(i).nonzero()[0] if self.cutting_plane_tree[node_index]['cutRHS'] != None else []# the set of constraints that contain the variable 'var'
                for j in constr_index:
                    eqConstraint.addTerms(self.cutting_plane_tree[node_index]['cutA'][j, i], cglp_gamma[node_index][j])

                cglp.addConstr(eqConstraint == 0, name=f'equation2_{node_index}_{var}')
        # normalization constraint
        if self.normalization == 'SNC':
            cglp.addConstr(normalizationConstraint == 0, name='normalizationConstraint')
        elif self.normalization == 'fix_pi0':
            cglp.addConstr(pi0 == 1, name='normalizationConstraint')

        # Set parameters
        cglp.Params.OutputFlag = self.cglp_OutputFlag
        cglp.Params.Threads = self.cglp_Threads
        cglp.Params.MIPGap = self.cglp_MIPGap
        cglp.Params.TimeLimit = self.cglp_TimeLimit
        cglp.Params.MIPFocus = self.cglp_MIPFocus  # what are you focus on? => 0: balanced, 1: feasible sol, 2: optimal sol, 3: bound, 4: hidden feasible sol, 5: hidden optimal sol

        cglp.update()
        # cglp.write(f'cglp_{self.iteration}.lp')
        cglp.optimize()
        if cglp.status == GRB.OPTIMAL:  #
            # add a cut to the LP relaxation model
            piBest = cglp.getAttr('x', pi)  # piBest is a dictionary
            pi0Best = cglp.getVarByName('pi_0').x  # pi0Best is a float
            # add the cut to the child_node LP relaxation model
            cut = {}
            cut['piBest'] = piBest
            cut['pi0Best'] = pi0Best
            self.branch_bound_tree[child_node_index]['cuts'][1] = cut
            self.nodal_problem(child_node_index)
        else:
            print(f'cglp status: {cglp.status}')
            return None


    def solve(self):
        time_init = time.time()
        while self.iteration <= self.maxIteration:
            iter_begin = time.time()
            # selection a node to branch
            if self.iteration == 0:
                self.branch_node_selection()
            if self.OPT == True or self.upper_bound['value'] - self.lower_bound['value'] <= 1e-1:
                self.print_iteration_info()
                return
            # selection a variable to branch
            self.branch_variable_selection()
            self.naive_cpt_variable_selection()
            # branch
            self.branching_and_cut_generation()
            # bound
            self.fathom_by_bounding()
            iter_end = time.time()
            overall = iter_end - time_init
            iteration_time = iter_end - iter_begin
            self.branch_node_selection()
            self.iteration += 1
            self.print_iteration_info(iteration_time, overall)
