from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB
import torch
import torch.nn as nn
import numpy as np
import os
import math
import time

class CuttingPlaneMethod:
    def __init__(self, instanceName, maxIteration = 100, OutputFlag = 0, Threads = 1, MIPGap = 0.0, TimeLimit = 3600, MIPFocus = 2, cglp_OutputFlag = 0, cglp_Threads = 1, cglp_MIPGap = 0.0, cglp_TimeLimit = 100, cglp_MIPFocus = 0, addCutToMIP = False, number_branch_var = 2, normalization = 'SNC'):
        self.iteration = 0
        self.maxIteration = maxIteration
        self.maxBound = 1e5
        self.OPT = False
        # Instance Info
        self.mipModel = None
        self.instanceName = instanceName
        self.A = None
        self.RHS = None
        self.LB = None
        self.UB = None
        self.variables = None
        self.varName = []
        self.integer_vars = None
        self.binary_vars = None                     # item: variables
        self.non_integer_vars = {}                  # iter: {varName: distance}
        self.non_binary_vars = {}
        self.lp_relaxation = None
        self.lp_sol = None
        self.varName_map_position = {}
        # Cut List
        self.normalization = normalization
        self.number_branch_var = number_branch_var
        self.nodeSet = {}                           # nodeSet[node] <- (LB, UB)
        self.addCutToMIP = addCutToMIP
        self.branchVar = {}							# branchVar[iter] <- var
        self.coefList = {}							# coeflist[iter] <- (subg, 1) or ( - piBest, pi0Best)
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
        # initialize NN parameters
        self.Cand = None    
         # intialize the instance
        self.readin()         

    def readin(self):
        #load instance info
        ins_dir=f'benchmark/' + self.instanceName + '.mps.gz'
        self.mipModel = gp.read(ins_dir)
        self.variables = self.mipModel.getVars()
        self.A = self.mipModel.getA()
        # self._A = deepcopy(self.mipModel.getA())
        self.RHS = self.mipModel.getAttr('RHS')
        self.SENSE = self.mipModel.getAttr('Sense')
        self.LB = self.mipModel.getAttr('LB')
        for i in range(len(self.LB)):
            if self.LB[i] == - math.inf:
                self.LB[i] = - self.maxBound
        self.UB = self.mipModel.getAttr('UB')
        for i in range(len(self.UB)):
            if self.UB[i] == math.inf:
                self.UB[i] = self.maxBound
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
        self.mipModel.Params.MIPFocus = self.MIPFocus             # what are you focus on? => 0: balanced, 1: feasible sol, 2: optimal sol, 3: bound, 4: hidden feasible sol, 5: hidden optimal sol
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
        self.lp_relaxation = self.mipModel.relax()
        self.lp_relaxation.update()
          
    
    def master_problem(self):
        # Create the LP relaxation model
        self.lp_relaxation.optimize()
        self.lp_obj_value[self.iteration] = self.lp_relaxation.objVal
        self.lp_sol = self.lp_relaxation.getAttr('x')
        # update the ineqConstraintaint information to the current LP relaxation with cuts
        self.A = self.lp_relaxation.getA()
        self.RHS = self.lp_relaxation.getAttr('RHS')
        self.SENSE = self.lp_relaxation.getAttr('Sense')

        if self.lp_relaxation.status == GRB.OPTIMAL:
            # check if the solution is integer
            self.Cand = []
            self.non_integer_vars[self.iteration] = {}
            self.non_binary_vars[self.iteration] = {}
            for v in self.integer_vars:
                relaxed_value = self.lp_relaxation.getVarByName(v.varName).x
                if not math.isclose(relaxed_value, round(relaxed_value), abs_tol=1e-6):
                    self.non_integer_vars[self.iteration][v.varName] = abs(relaxed_value - round(relaxed_value))
                    self.Cand.append(v.varName)
            for v in self.binary_vars:
                relaxed_value = self.lp_relaxation.getVarByName(v.varName).x
                if not math.isclose(relaxed_value, round(relaxed_value)):
                    self.non_binary_vars[self.iteration][v.varName] = abs(relaxed_value - round(relaxed_value))
                    self.Cand.append(v.varName)
            if len(self.non_integer_vars[self.iteration]) == 0 and len(self.non_binary_vars[self.iteration]) == 0:
                self.OPT = True

        self.iteration += 1

    def cut_generation(self):
        cglp = gp.Model("cglp")
        # Create variables
        pi = cglp.addVars(self.varName,vtype=GRB.CONTINUOUS,lb=-float('inf'),name=f"pi",obj=0.0)
        pi0 = cglp.addVar(vtype=GRB.CONTINUOUS,lb=-float('inf'),name=f'pi_0',obj=-1.0)
        cglp.update()
        # Set objective
        cglp.setObjective(gp.quicksum(pi[v] * self.lp_relaxation.getVarByName(v).x for v in self.varName) - pi0, GRB.MINIMIZE)

        num_constrs, num_vars =self.A.shape[0], self.A.shape[1] # self.mipModel.getAttr('NumConstrs'), self.mipModel.getAttr('NumVars')
        cglp_lambda = {}
        cglp_mu = {}
        cglp_v = {}
        # cglp normalization constraint
        normalizationConstraint=gp.LinExpr(-1.0)
        for node_index, node in self.nodeSet.items():
            cglp_lambda[node_index] = []
            cglp_mu[node_index] = []
            cglp_v[node_index] = []
            # cglp equation (3) in ORL paper
            ineqConstraint=gp.LinExpr(-pi0)
            for i in range(num_constrs):
                sense = self.SENSE[i]
                if sense == '<':
                    cglp_lambda[node_index].append(cglp.addVar(vtype=GRB.CONTINUOUS,lb=0.0,name=f'lambda_{node_index}_{i}',obj=0.0))
                    ineqConstraint.addTerms(-self.RHS[i], cglp_lambda[node_index][i])
                    normalizationConstraint.addTerms(1, cglp_lambda[node_index][i])
                elif sense == '>':
                    cglp_lambda[node_index].append(cglp.addVar(vtype=GRB.CONTINUOUS,lb=0.0,name=f'lambda_{node_index}_{i}',obj=0.0))
                    ineqConstraint.addTerms(self.RHS[i], cglp_lambda[node_index][i])
                    normalizationConstraint.addTerms(1, cglp_lambda[node_index][i])
                elif sense == '=':
                    # Ax = b <=> Ax >= b and -Ax >= -b
                    cglp_lambda[node_index].append(cglp.addVars(['+', '-'], vtype=GRB.CONTINUOUS,lb=0.0,name=f'lambda_{node_index}_{i}',obj=0.0))
                    ineqConstraint.addTerms(self.RHS[i], cglp_lambda[node_index][i]['+'])
                    normalizationConstraint.addTerms(1, cglp_lambda[node_index][i]['+'])

                    ineqConstraint.addTerms(-self.RHS[i], cglp_lambda[node_index][i]['-'])
                    normalizationConstraint.addTerms(1, cglp_lambda[node_index][i]['-'])
                
            for i in range(num_vars):
                var = self.varName[i]
                cglp_mu[node_index].append(cglp.addVar(vtype=GRB.CONTINUOUS,lb=0.0,name=f'mu_{node_index}_{var}',obj=0.0))
                cglp_v[node_index].append(cglp.addVar(vtype=GRB.CONTINUOUS,lb=0.0,name=f'v_{node_index}_{var}',obj=0.0))
                ineqConstraint.addTerms(node['LB'][i], cglp_mu[node_index][i])
                ineqConstraint.addTerms(-node['UB'][i], cglp_v[node_index][i])
                normalizationConstraint.addTerms(1, cglp_mu[node_index][i])
                normalizationConstraint.addTerms(1, cglp_v[node_index][i])

            cglp.addConstr(ineqConstraint >= 0, name=f'equation3_{node_index}')

            # cglp equation (2) in ORL paper
            for i in range(num_vars): 
                var = self.varName[i]
                eqConstraint=gp.LinExpr(-pi[var])
                eqConstraint.addTerms(1, cglp_mu[node_index][i])
                eqConstraint.addTerms(-1, cglp_v[node_index][i])

                ## add matrix multiplication term
                constr_index = self.A.getcol(i).nonzero()[0] # the set of constraints that contain the variable 'var'
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
        cglp.Params.MIPFocus = self.cglp_MIPFocus             # what are you focus on? => 0: balanced, 1: feasible sol, 2: optimal sol, 3: bound, 4: hidden feasible sol, 5: hidden optimal sol


        cglp.update()
        # cglp.write(f'cglp_{self.iteration}')
        cglp.optimize()
        if cglp.status == GRB.OPTIMAL: # 
            # add a cut to the LP relaxation model
            #TODO:: What is the difference between addConstr and cbCut? Could I use cbCut to add a cut to the LP relaxation model?
            piBest = cglp.getAttr('x', pi) # piBest is a dictionary
            pi0Best = cglp.getVarByName('pi_0').x # pi0Best is a float

            newCut=gp.LinExpr(- pi0Best) 
            for var in self.varName:
                newCut += self.lp_relaxation.getVarByName(var) * piBest[var]
            self.lp_relaxation.addConstr(newCut>=0.0, name=f'cut_{self.iteration-1}')
            self.lp_relaxation.update()
            if self.addCutToMIP:
                self.mipModel.addConstr(newCut>=0.0, name=f'cut_{self.iteration-1}')
                self.mipModel.update()
                
            # add a cut to the cut list
            self.coefList[self.iteration-1] = {}
            self.coefList[self.iteration-1]['piBest'] = piBest
            self.coefList[self.iteration-1]['pi0Best'] = pi0Best            
        else:
            print(f'cglp status: {cglp.status}')
            return None

    def print_iteration_info(self, cut_time = 0.0, iteration_time = 0.0, overall = 0.0):
        if self.OPT == True:
            print(f'---------------------------------------------------------------------------------------------------------------------------------')
            print('Optimality of MIP has been established!')
        if self.iteration == 1:
            print(f'This problem has {len(self.integer_vars)} integer variables and {len(self.binary_vars)} binary variables.')
            print(f'The optimal value of LP relaxation is {self.lp_obj_value[self.iteration-1]}.')
            print(f'---------------------------------------------------------------------------------------------------------------------------------')
            print(f'|  Iter  |  # fractional var  |  current value  |  Relative Improvement  |  Overall Improvement  |  Iter Time  |  Overall Time  |')
            print(f'---------------------------------------------------------------------------------------------------------------------------------')
        else:
            print('| '+'{:7d}'.format(self.iteration-1)+'| '+'{:19d}'.format(len(self.non_integer_vars[self.iteration-1]) + len(self.non_binary_vars[self.iteration-1]))+'| '+'{:16.4f}'.format(self.lp_obj_value[self.iteration-1])+'| '+'{:22.4f}'.format( abs(self.lp_obj_value[self.iteration-1]-self.lp_obj_value[self.iteration-2])/self.lp_obj_value[self.iteration-1] * 100 )+' | '+'{:21.4f}'.format(abs(self.lp_obj_value[self.iteration-1]-self.lp_obj_value[0])/self.lp_obj_value[0] * 100)+' | '+'{:11.4f}'.format(iteration_time)+' | '+'{:14.4f}'.format(overall)+' |')
        
        if self.iteration > self.maxIteration:
            print(f'---------------------------------------------------------------------------------------------------------------------------------')
              
    