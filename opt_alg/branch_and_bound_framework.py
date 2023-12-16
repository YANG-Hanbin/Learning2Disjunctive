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


class BranchAndBoundFramework:
    def __init__(self, instanceName, maxIteration=100, OutputFlag=0, Threads=1, MIPGap=0.0, TimeLimit=3600, MIPFocus=2,
                 cglp_OutputFlag=0, cglp_Threads=1, cglp_MIPGap=0.0, cglp_TimeLimit=100, cglp_MIPFocus=0,
                 addCutToMIP=False, number_branch_var=2, normalization='SNC'):
        self.iteration = 0
        self.maxIteration = maxIteration
        self.maxBound = 1e5
        self.OPT = False  # global optimality
        # Instance Info
        self.mipModel = None
        self.instanceName = instanceName
        self.A = None
        self.RHS = None
        self.LB = None
        self.UB = None
        self.modelSense = None
        self.variables = None
        self.varName = []
        self.integer_vars = None
        self.binary_vars = None  # item: variables
        self.non_integer_vars = {}  # iter: {varName: distance}
        self.non_binary_vars = {}
        self.lp_relaxation = None
        self.varName_map_position = {}
        # Cut List
        self.normalization = normalization
        self.number_branch_var = number_branch_var
        self.nodeSet = {}  # nodeSet[node] <- (LB, UB)
        self.addCutToMIP = addCutToMIP
        self.branchVar = {}  # branchVar[iter] <- var
        self.coefList = {}  # coeflist[iter] <- (subg, 1) or ( - piBest, pi0Best)
        # Records
        self.lp_obj_value = {}
        # Gurobi Model Info
        self.OutputFlag = OutputFlag
        self.Threads = Threads
        self.MIPGap = MIPGap
        self.TimeLimit = TimeLimit
        self.MIPFocus = MIPFocus
        self.cglp_OutputFlag = cglp_OutputFlag
        self.cglp_Threads = cglp_Threads
        self.cglp_MIPGap = cglp_MIPGap
        self.cglp_TimeLimit = cglp_TimeLimit
        self.cglp_MIPFocus = cglp_MIPFocus
        # Branch and Bound Tree Info
        self.branch_bound_tree = {}  # 1. key: node_index <- value: (trace, cuts, subproblem's optimal sol and val); 2. we only record the leaf nodes
        self.subproblem = None
        self.branch_node = None  # we will do branching on variable self.branch_variable (varName) in the node self.branch_node
        self.branch_variable = None  # varName
        self.lower_bound = {'node': 0, 'value': - math.inf}  #
        self.upper_bound = {'node': 0, 'value': math.inf}
        self.lower_bound_sol = None  # (for minimization problem) the nodal solution corresponding to the lower bound in the branch-and-bound tree
        self.incumbent = None
        # intialize the instance
        self.readin()

    def readin(self):
        # load instance info
        ins_dir = f'benchmark/' + self.instanceName + '.mps.gz'
        self.mipModel = gp.read(ins_dir)
        self.variables = self.mipModel.getVars()
        self.integer_vars = [var for var in self.variables if var.vType == GRB.INTEGER]
        self.binary_vars = [var for var in self.variables if var.vType == GRB.BINARY]

        position = 0
        for var in self.variables:
            self.varName.append(var.varName)
            self.varName_map_position[var.varName] = position
            position += 1

        # Set parameters
        self.mipModel.Params.OutputFlag = self.OutputFlag
        self.mipModel.Params.LogToConsole = 0
        self.mipModel.Params.Threads = self.Threads
        self.mipModel.Params.MIPGap = self.MIPGap
        self.mipModel.Params.TimeLimit = self.TimeLimit
        # self.mipModel.Params.Cuts = 0                 # whether use cuts => 0: no cut, 1: auto, 2: conservative, 3: aggressive
        # self.mipModel.Params.Heuristics = 0           # whether use heuristics => 0: no heuristic, 1: auto, 2: conservative, 3: aggressive
        # self.mipModel.Params.Presolve = 0             # whether use presolve => 0: no presolve, 1: presolve, 2: aggressive presolve
        # self.mipModel.Params.Method = 0               # how to solve LP => 0: primal simplex, 1: dual simplex, 2: barrier, 3: concurrent
        # self.mipModel.Params.Crossover = 0            # whether use crossover => -1: auto, 0: no crossover, 1: primal crossover, 2: dual crossover
        self.mipModel.Params.MIPFocus = self.MIPFocus  # what are you focus on? => 0: balanced, 1: feasible sol, 2: optimal sol, 3: bound, 4: hidden feasible sol, 5: hidden optimal sol
        # self.mipModel.Params.LazyineqConstraintaints = 1      # whether use lazy ineqConstraintaints => 0: no lazy ineqConstraintaints, 1: lazy ineqConstraintaints
        # self.mipModel.Params.CutsFactor = 1
        # self.mipModel.Params.CliqueCuts = 0
        # self.mipModel.Params.CoverCuts = 0
        # self.mipModel.Params.FlowCoverCuts = 0
        # self.mipModel.Params.FlowPathCuts = 0
        # self.mipModel.Params.GUBCoverCuts = 0
        # self.mipModel.Params.ImpliedCuts = 0
        # self.mipModel.Params.InfProofCuts = 0
        # self.mipModel.Params.MIPSepCuts = 0
        # self.mipModel.Params.MIRCutCuts = 0
        # self.mipModel.Params.ModKCuts = 0
        # self.mipModel.Params.NetworkCuts = 0
        # self.mipModel.Params.PathCutCuts = 0
        # self.mipModel.Params.ProjectedCGCuts = 0
        # self.mipModel.Params.RLTFCuts = 0             # 0: no cut, 1: auto, 2: conservative, 3: aggressive
        # self.mipModel.Params.StrongCGCuts = 0
        # self.mipModel.Params.PrePasses = 0 # the number of presolve times => 0: auto, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5

        # get the LP relaxation problem
        self.lp_relaxation = self.mipModel.relax()
        self.lp_relaxation.update()
        self.variables = self.lp_relaxation.getVars()
        self.A = self.lp_relaxation.getA()
        self.RHS = self.lp_relaxation.getAttr('RHS')
        self.SENSE = self.lp_relaxation.getAttr('Sense')
        self.modelSense = 'min' if self.mipModel.ModelSense == 1 else 'max'
        self.LB = self.lp_relaxation.getAttr('LB')
        for i in range(len(self.LB)):
            if self.LB[i] == - math.inf:
                self.LB[i] = - self.maxBound
        self.UB = self.lp_relaxation.getAttr('UB')
        for i in range(len(self.UB)):
            if self.UB[i] == math.inf:
                self.UB[i] = self.maxBound
        node = {}
        self.lp_relaxation.optimize()
        if self.lp_relaxation.status == GRB.OPTIMAL:
            if self.modelSense == 'min':
                self.lower_bound['value'] = self.lp_relaxation.objVal
            else:
                self.upper_bound['value'] = self.lp_relaxation.objVal

            node['sol'] = self.lp_relaxation.x
            node['value'] = self.lp_relaxation.objVal
            node['cuts'] = {}  # record all additional cuts
            node['trace'] = []  # record all additional bounding constraints

            non_integer_vars = {}  # the sef of int varName that are fractional
            non_binary_vars = {}
            # check if the solution is integer
            for v in self.integer_vars:
                relaxed_value = self.lp_relaxation.getVarByName(v.varName).x
                if not math.isclose(relaxed_value, round(relaxed_value), abs_tol=1e-6):
                    non_integer_vars[v.varName] = abs(relaxed_value - round(relaxed_value))
            for v in self.binary_vars:
                relaxed_value = self.lp_relaxation.getVarByName(v.varName).x
                if not math.isclose(relaxed_value, round(relaxed_value)):
                    non_binary_vars[v.varName] = abs(relaxed_value - round(relaxed_value))
        node['fractional_int'] = non_integer_vars
        node['fractional_bin'] = non_binary_vars
        # update the nodal info
        self.branch_bound_tree[0] = node
        if len(non_integer_vars) + len(non_binary_vars) == 0:
            self.OPT = True

    def branch_node_selection(self):
        # update self.branch_node in self.branch_bound_tree

        if len(self.branch_bound_tree.keys()) == 1:
            self.branch_node = list(self.branch_bound_tree.keys())[0]
            self.lower_bound_sol = self.branch_bound_tree[self.branch_node]['sol']
        else:
            # The first rule -- Best Bound Rule: for min problem, choose the node with the smallest lower bound; for
            # max problem, choose the node with the largest upper bound
            self.branch_node = None
            lower_bound = math.inf
            upper_bound = -math.inf
            if self.modelSense == 'min':
                for node_index, node in self.branch_bound_tree.items():
                    if node['value'] <= lower_bound:
                        self.branch_node = node_index
                        lower_bound = node['value']
                        self.lower_bound = {'node': node_index, 'value': lower_bound}
                        self.lower_bound_sol = node['sol']
            else:
                for node_index, node in self.branch_bound_tree.items():
                    if node['value'] >= upper_bound:
                        self.branch_node = node_index
                        upper_bound = node['value']
                        self.upper_bound = {'node': node_index, 'value': upper_bound}
                        self.lower_bound_sol = node['sol']
            # TODO:: The second rule -- Deepest Node First Rule
            # self.branch_node = max(self.branch_bound_tree.keys())

    def branch_variable_selection(self):
        # TODO:: add ML model to choose the variable to branch
        # according to self.branch_node, choose a self.branch_variable 
        # choose the variable to branch: Maximum Fractionality Rule
        node = self.branch_bound_tree[self.branch_node]
        number_of_noninteger = len(node['fractional_int'])
        number_of_nonbinary = len(node['fractional_bin'])

        maxKey = None
        maxDistance = None
        if number_of_noninteger > 0:
            maxKey = max(node['fractional_int'], key=node[
                'fractional_int'].get)  # find the integer variables that have the largest distance to the nearest integer
            maxDistance = node['fractional_int'][maxKey]

        if number_of_nonbinary > 0:
            tmp_maxKey = max(node['fractional_bin'], key=node[
                'fractional_bin'].get)  # find the integer variables that have the largest distance to the nearest integer
            tmp_maxDistance = node['fractional_bin'][tmp_maxKey]
            if maxDistance == None or tmp_maxDistance > maxDistance:
                maxKey = tmp_maxKey
                maxDistance = tmp_maxDistance
        self.branch_variable = maxKey

    def branching(self):
        # create two new nodes with info in self.branch_node, del self.branch_node, update two new nodes with the
        # method self.nodal_problem
        node = self.branch_bound_tree[self.branch_node]  # father node
        pos = self.varName_map_position[self.branch_variable]

        left_node = {}
        left_node['cuts'] = deepcopy(node['cuts'])
        left_node['trace'] = deepcopy(node['trace'])
        left_node['trace'].append([self.branch_variable, '<', math.floor(node['sol'][pos])])  # x <= floor(xhat)

        right_node = {}
        right_node['cuts'] = deepcopy(node['cuts'])
        right_node['trace'] = deepcopy(node['trace'])
        right_node['trace'].append([self.branch_variable, '>', math.ceil(node['sol'][pos])])  # x >= ceil(xhat)

        left_node_ind = max(self.branch_bound_tree.keys()) + 1
        right_node_ind = left_node_ind + 1
        self.branch_bound_tree[left_node_ind] = left_node
        self.branch_bound_tree[right_node_ind] = right_node
        del self.branch_bound_tree[self.branch_node]
        self.nodal_problem(left_node_ind)
        self.nodal_problem(right_node_ind)

    def fathom_by_bounding(self):
        # fathom nodes
        if self.modelSense == 'min':
            upper_bound = self.upper_bound['value']
            if upper_bound == math.inf:
                return
            else:
                to_delete = []
                for node_index, node in self.branch_bound_tree.items():
                    if node['value'] > upper_bound:
                        to_delete.append(node_index)
                for node_index in to_delete:
                    del self.branch_bound_tree[node_index]
                    print(f'fathom node {node_index} by bounding')
        else:
            lower_bound = self.lower_bound['value']
            if lower_bound == -math.inf:
                return
            else:
                to_delete = []
                for node_index, node in self.branch_bound_tree.items():
                    if node['value'] < lower_bound:
                        to_delete.append(node_index)
                    for node_index in to_delete:
                        del self.branch_bound_tree[node_index]
                        print(f'fathom node {node_index} by bounding')

    def nodal_problem(self, node_index):  # node_index is a new child node
        # Create the nodal subproblem model
        self.subproblem = self.lp_relaxation.copy()
        node = self.branch_bound_tree[node_index]
        cutSet = node['cuts']  # the set of cuts inherited from its parent's node
        trace = node['trace']  # the set of additional bound constraint along the branch-and-bound tree
        # add bounds to subproblem
        for item in trace:
            varName, sense, bound = item
            if sense == '<':
                self.subproblem.getVarByName(varName).ub = bound  # x <= floor(xhat) = ub
            elif sense == '>':
                self.subproblem.getVarByName(varName).lb = bound  # x >= ceil(xhat) = lb
            elif sense == '=':  # this is useless
                self.subproblem.getVarByName(varName).ub = bound
                self.subproblem.getVarByName(varName).lb = bound
        # add cuts to subproblem
        for i, cut in cutSet.items():
            piBest = cut['piBest']  # piBest is a dictionary
            pi0Best = cut['pi0Best']  # pi0Best is a float
            newCut = gp.LinExpr(- pi0Best)
            for var in self.varName:
                newCut += self.subproblem.getVarByName(var) * piBest[var]
            self.subproblem.addConstr(newCut >= 0.0)

            # cutLinExpr = cut['LinExpr']
            # self.subproblem.addConstr(cutLinExpr>=0.0)

        # TODO:: Check the feasibility of the following cut-appending way
        # for i, cut in cutSet.items():
        #     self.subproblem.addConstr(newCut>=0.0)
        self.subproblem.update()
        self.subproblem.optimize()
        if self.subproblem.status == GRB.INFEASIBLE:
            # fathom infeasible nodes
            del self.branch_bound_tree[node_index]
            print(f'fathom node {node_index} by infeasibility')
            return

        node['sol'] = self.subproblem.x
        node['value'] = self.subproblem.objVal

        non_integer_vars = {}
        non_binary_vars = {}
        if self.subproblem.status == GRB.OPTIMAL:
            # check if the solution is integer
            for v in self.integer_vars:
                relaxed_value = self.subproblem.getVarByName(v.varName).x
                if not math.isclose(relaxed_value, round(relaxed_value), abs_tol=1e-6):
                    non_integer_vars[v.varName] = abs(relaxed_value - round(relaxed_value))
            for v in self.binary_vars:
                relaxed_value = self.subproblem.getVarByName(v.varName).x
                if not math.isclose(relaxed_value, round(relaxed_value)):
                    non_binary_vars[v.varName] = abs(relaxed_value - round(relaxed_value))

        node['fractional_int'] = non_integer_vars
        node['fractional_bin'] = non_binary_vars
        # update the nodal info
        self.branch_bound_tree[node_index] = node
        # self.iteration += 1
        if len(non_integer_vars) + len(non_binary_vars) == 0:
            # find a feasible sol
            if self.modelSense == 'min':
                if node['value'] <= self.upper_bound['value']:
                    self.incumbent = node['sol']
                    self.upper_bound['node'] = node_index
                    self.upper_bound['value'] = node['value']
            else:
                if node['value'] >= self.lower_bound['value']:
                    self.incumbent = node['sol']
                    self.lower_bound['node'] = node_index
                    self.lower_bound['value'] = node['value']

    def cut_generation_node_selection(self):
        # TODO:: Using RL to selection a set of nodes to generate cuts
        # here we use up to 3 most promising nodes to generate cut
        node_values = [(node_index, node['value']) for node_index, node in self.branch_bound_tree.items()]
        node_values.sort(key=lambda x: x[1])
        nodeSet = [node_index for node_index, value in node_values[:3]]
        # nodeSet = self.branch_bound_tree.keys() 

        self.nodeSet = {}
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

    def cut_generation(self):
        cglp = gp.Model("cglp")
        # Create variables
        pi = cglp.addVars(self.varName, vtype=GRB.CONTINUOUS, lb=-float('inf'), name=f"pi", obj=0.0)
        pi0 = cglp.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), name=f'pi_0', obj=-1.0)
        cglp.update()
        # Set objective
        if self.incumbent == None:
            # cglp.setObjective(0, GRB.MINIMIZE)
            cglp.setObjective(
                gp.quicksum(pi[v] * self.lower_bound_sol[self.varName_map_position[v]] for v in self.varName) - pi0,
                GRB.MINIMIZE)
        else:
            cglp.setObjective(
                gp.quicksum(pi[v] * self.incumbent[self.varName_map_position[v]] for v in self.varName) - pi0,
                GRB.MINIMIZE)

        num_constrs, num_vars = self.A.shape[0], self.A.shape[
            1]  # self.mipModel.getAttr('NumConstrs'), self.mipModel.getAttr('NumVars')
        cglp_lambda = {}
        cglp_mu = {}
        cglp_v = {}
        # cglp normalization constraint
        normalizationConstraint = gp.LinExpr(-1.0)

        for node_index, node in self.nodeSet.items():
            cglp_lambda[node_index] = []
            cglp_mu[node_index] = []
            cglp_v[node_index] = []
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

            cglp.addConstr(ineqConstraint >= 0, name=f'equation3_{node_index}')

            # cglp equation (2) in ORL paper
            for i in range(num_vars):
                var = self.varName[i]
                eqConstraint = gp.LinExpr(-pi[var])
                eqConstraint.addTerms(1, cglp_mu[node_index][i])
                eqConstraint.addTerms(-1, cglp_v[node_index][i])

                ## add matrix multiplication term
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
            # TODO:: What is the difference between addConstr and cbCut? Could I use cbCut to add a cut to the LP relaxation model?
            piBest = cglp.getAttr('x', pi)  # piBest is a dictionary
            pi0Best = cglp.getVarByName('pi_0').x  # pi0Best is a float
            for node_index in self.nodeSet.keys():
                cut = {}
                cut['piBest'] = piBest
                cut['pi0Best'] = pi0Best
                self.branch_bound_tree[node_index]['cuts'][len(self.branch_bound_tree[node_index]['cuts']) + 1] = cut
                self.nodal_problem(node_index)

            # newCut=gp.LinExpr(- pi0Best) 
            # for var in self.varName:
            #     newCut += self.lp_relaxation.getVarByName(var) * piBest[var]
        else:
            print(f'cglp status: {cglp.status}')
            return None

    def print_iteration_info(self, cut_time=0.0, iteration_time=0.0, overall=0.0):
        if self.OPT == True:
            print(f'---------------------------------------------------------------------------------------------------------')
            print('Optimality of MIP has been established!')
        if self.iteration >= 1:
            if self.iteration == 1:
                print(f'This problem has {len(self.integer_vars)} integer variables and {len(self.binary_vars)} binary variables.')
                print(f'The optimal value of LP relaxation is {self.lp_relaxation.objVal}.')
                print(f'---------------------------------------------------------------------------------------------------------')
                print(f'|  Iter  |    Lower Bound   |   Overall Improvement   |    Upper Bound   |  Iter Time  |  Overall Time  |')
                print(f'---------------------------------------------------------------------------------------------------------')
            else:
                print('| ' + '{:7d}'.format(self.iteration - 1) +
                      '| ' + '{:17.4f}'.format(self.lower_bound['value']) +
                      '| ' + '{:23.4f}'.format(
                    abs(self.lower_bound['value'] - self.lp_relaxation.objVal) / self.lp_relaxation.objVal * 100) +
                      ' | ' + '{:16.4f}'.format(self.upper_bound['value']) +
                      ' | ' + '{:11.4f}'.format(iteration_time) + ' | ' +
                      '{:14.4f}'.format(overall) + ' |')

        if self.iteration > self.maxIteration:
            print(f'---------------------------------------------------------------------------------------------------------')

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
            # branch
            self.branching()
            # bound
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
