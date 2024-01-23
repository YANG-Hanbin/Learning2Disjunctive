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
from torch.distributions import Categorical

from opt_alg.branch_and_bound_framework import BranchAndBoundFramework
from rl_alg.model import var_sorter, var_decider, SharedAdam, Critic

"""
    The class RLBranchAndCuttingPlaneTreeAlgorithm is a subclass of BranchAndBoundFramework, which implements the RL based branch-and-CPT algorithm.
    - It allows to select multiple branching variables instead of one branching variable in each iteration.
"""


class RLBranchAndCuttingPlaneTreeAlgorithm(BranchAndBoundFramework):
    def __init__(self, *args, number_branch_var=2, number_branch_node=1, cptVariableSelectionMode='MFR',
                 disjunctiveCut=True, maxBuffer=5, lock=None, cglp_Threads=8, global_model=None, **kwargs):
        # Initialize parent class with all arguments passed
        super().__init__(*args, **kwargs)
        # Initialization for new attributes
        self.number_branch_var = number_branch_var
        self.number_branch_node = number_branch_node
        self.cutting_plane_tree = {}  # key: node_index <- value: (trace, cuts), all leaf nodes
        self.nodeSet = {}  # nodeSet[node] <- (LB, UB)
        self.cptVariableSelectionMode = cptVariableSelectionMode  # 'MFR', 'RAND' in CPT
        self.disjunctiveCut = disjunctiveCut
        self.cglp_Threads = cglp_Threads

        ## RL part
        self.train = True
        self.global_model = global_model
        self.lock = lock
        self.RLmodel = var_sorter(v_size=6,
                                  c_size=3, sample_sizes=[64, 128], multi_head=2, natt=2)  # model for cut var selection
        self.RLmodel_branch = var_decider(v_size=6,
                                          c_size=3, sample_sizes=[64, 128], multi_head=2,
                                          natt=2)  # model for branch var selection
        self.critic = Critic(input_size=128 * 3, sample_sizes=[64, 64])

        self.optimizer = torch.optim.Adam(
            list(self.RLmodel.parameters()) + list(self.RLmodel_branch.parameters()) + list(self.critic.parameters()))
        # key idea: use local ac to obtain grad, then apply to global

        self.last_bound = None

        self.maxBuffer = maxBuffer

        self.tensorA = None
        self.col_feature = None
        self.row_feature = None

        self.go_next_sampled = True
        self.go_next = True

        self.buffer_length = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.predVals = []
        self.predProb1 = []
        self.predProb2 = []
        self.predAct1 = []
        self.predAct2 = []
        self.predProbDec = []
        self.predActDec = []

        self.gamma = 0.999

        self.critic_criteria = nn.MSELoss()
        self.critic_buffer = None

    def A_to_sparse_tensor(self):
        self.tensorA = self.A.tocoo()
        indices = np.vstack((self.tensorA.row, self.tensorA.col))
        data = self.tensorA.data
        self.tensorA = torch.sparse_coo_tensor(indices, data, self.tensorA.shape, dtype=torch.float32)

    def clearMemory(self):
        self.buffer_length = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.predVals = []
        self.predProb1 = []
        self.predProb2 = []
        self.predAct1 = []
        self.predAct2 = []
        self.predProbDec = []
        self.predActDec = []

    def learnFromBuffer_global(self):
        # Calculate reward
        predValue = self.predVals[-1]
        bat_reward = []
        for reward in self.rewards[::-1]:
            predValue *= self.gamma
            predValue += reward
            bat_reward.append(predValue)
        bat_reward.reverse()

        bat_reward = torch.tensor(bat_reward)
        self.predVals = torch.tensor(self.predVals)
        # Get all losses
        critic_loss = self.critic_criteria(bat_reward, self.predVals)

        loss1 = None
        loss2 = None
        for i in range(len(self.predProb1)):
            dist1 = Categorical(self.predProb1[i])
            dist2 = Categorical(self.predProb2[i])
            act1 = torch.tensor(self.predAct1[i])
            act2 = torch.tensor(self.predAct2[i])
            logProb1 = dist1.log_prob(act1)
            logProb2 = dist2.log_prob(act2)
            if loss1 is None:
                loss1 = -(logProb1 * (bat_reward - self.predVals)[i]).mean()
            else:
                loss1 += -(logProb1 * (bat_reward - self.predVals)[i]).mean()
            if loss2 is None:
                loss2 = -(logProb2 * (bat_reward - self.predVals)[i]).mean()
            else:
                loss2 += -(logProb2 * (bat_reward - self.predVals)[i]).mean()
        total_loss = (loss1 + loss2 + critic_loss).mean()
        for local_p, global_p in zip(self.RLmodel, self.global_model[0]):
            global_p._grad = local_p.grad
        for local_p, global_p in zip(self.RLmodel_branch, self.global_model[1]):
            global_p._grad = local_p.grad
        for local_p, global_p in zip(self.critic, self.global_model[2]):
            global_p._grad = local_p.grad

        self.global_model[3].zero_grad()
        total_loss.backward()
        self.global_model[3].step()

        self.RLmodel.load_state_dict(self.global_model[0].state_dict())
        self.RLmodel_branch.load_state_dict(self.global_model[1].state_dict())
        self.critic.load_state_dict(self.global_model[2].state_dict())

        # Clear memory buffer to init
        self.clearMemory()

        state = {'model': self.RLmodel.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(state, f'./models/ckp_RLmodel_{self.iteration}.mdl')
        state2 = {'model': self.RLmodel_branch.state_dict()}
        torch.save(state2, f'./models/ckp_RLmodel_branch_{self.iteration}.mdl')
        state3 = {'model': self.critic.state_dict()}
        torch.save(state3, f'./models/ckp_critic_{self.iteration}.mdl')
        print('ModelSaved!!!!')

    def learnFromBuffer(self):
        # Calculate reward
        predValue = self.predVals[-1]
        bat_reward = []
        for reward in self.rewards[::-1]:
            predValue *= self.gamma
            predValue += reward
            bat_reward.append(predValue)
        bat_reward.reverse()

        bat_reward = torch.tensor(bat_reward)
        self.predVals = torch.tensor(self.predVals)
        # Get all losses
        critic_loss = self.critic_criteria(bat_reward, self.predVals)

        loss1 = None
        loss2 = None
        for i in range(len(self.predProb1)):
            dist1 = Categorical(self.predProb1[i])
            dist2 = Categorical(self.predProb2[i])
            act1 = torch.tensor(self.predAct1[i])
            act2 = torch.tensor(self.predAct2[i])
            logProb1 = dist1.log_prob(act1)
            logProb2 = dist2.log_prob(act2)
            if loss1 is None:
                loss1 = -(logProb1 * (bat_reward - self.predVals)[i]).mean()
            else:
                loss1 += -(logProb1 * (bat_reward - self.predVals)[i]).mean()
            if loss2 is None:
                loss2 = -(logProb2 * (bat_reward - self.predVals)[i]).mean()
            else:
                loss2 += -(logProb2 * (bat_reward - self.predVals)[i]).mean()
        # dist1 = Categorical(self.predProb1)
        # dist2 = Categorical(self.predProb2)
        # logProb1 = dist1.log_prob(self.predAct1)
        # logProb2 = dist2.log_prob(self.predAct2)
        # loss1 = -logProb1*(bat_reward - self.predVals)
        # loss2 = -logProb2*(bat_reward - self.predVals)
        total_loss = (loss1 + loss2 + critic_loss).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Clear memory buffer to init
        self.clearMemory()

        state = {'model': self.RLmodel.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(state, f'./models/ckp_RLmodel_{self.iteration}.mdl')
        state2 = {'model': self.RLmodel_branch.state_dict()}
        torch.save(state2, f'./models/ckp_RLmodel_branch_{self.iteration}.mdl')
        state3 = {'model': self.critic.state_dict()}
        torch.save(state3, f'./models/ckp_critic_{self.iteration}.mdl')
        print('ModelSaved!!!!')

    def load_from_checkpoint(self, ckp_idx):
        # Load RL model from existing checkpoints with string name
        cpu_dev = torch.device('cpu')
        dct = torch.load(f'./models/ckp_RLmodel_{ckp_idx}.mdl', map_location=cpu_dev)
        self.RLmodel.load_state_dict(dct['model'])
        dct = torch.load(f'./models/ckp_RLmodel_branch_{ckp_idx}.mdl', map_location=cpu_dev)
        self.RLmodel_branch.load_state_dict(dct['model'])
        dct = torch.load(f'./models/ckp_critic_{ckp_idx}.mdl', map_location=cpu_dev)
        self.critic.load_state_dict(dct['model'])
        print('Checkpoint loaded')

    def query_state(self, node, candidates):
        # A matrix update
        self.A_to_sparse_tensor()

        # variable feature update
        col_feat = [None] * len(self.variables)
        for var in self.variables:
            rel_sol_indx = self.varName_map_position[var.VarName]
            col_value = node['sol'][rel_sol_indx]
            # if abs(col_val-round(col_val,0))>1e-6 and (vtp=='B' or vtp=='I'):
            #     cand.append(col_var.VarName)

            # compose features
            isBin = 0
            isInt = 0
            if var.VType == 'I':
                isInt = 1
            elif var.VType == 'B':
                isBin = 1
            # ---- [lp_sol, LB, UB, isBin, isInt, reduced_cost, lp_obj]
            col_feat[rel_sol_indx] = [col_value, self.LB[rel_sol_indx], self.UB[rel_sol_indx], isBin, isInt, var.Obj]
        col_feat = torch.as_tensor(col_feat, dtype=torch.float32)

        # constraint feature update
        row_index_map = {}
        rcounter = 0
        row_feat = []
        for idx, constr in enumerate(self.lp_relaxation.getConstrs()):
            if constr.ConstrName not in row_index_map:
                row_index_map[constr.ConstrName] = rcounter
                rcounter += 1
            # compose row feature
            sense1 = 0
            sense2 = 0  # 01:leq 00:eq 10:geq
            if constr.Sense == '<':
                sense2 = 1
            elif constr.Sense == '>':
                sense1 = 1
            row_feat.append([constr.RHS, sense1, sense2])  # [RHS, SENSE]
        row_feat = torch.as_tensor(row_feat, dtype=torch.float32)
        cand = []
        names = []
        for vn in candidates:
            cand.append(self.varName_map_position[vn])
            names.append(vn)
        cand = torch.as_tensor(cand)
        cand = [cand, names]
        return [col_feat, row_feat, row_index_map, cand]

    def sample_disjunction_action(self, state, candidate, training=True):
        # Given action space and current space, sample an action
        cand = state[-1]
        cand_indx = cand[0]
        cand_names = cand[1]
        res = {}
        logits, tmp_buffer = self.RLmodel(self.tensorA, state[0], state[1])
        self.critic_buffer = torch.cat((self.critic_buffer, tmp_buffer), 1)
        if self.number_branch_var <= len(candidate):
            logits_sct = logits[cand_indx]
            index = logits_sct.multinomial(num_samples=self.number_branch_var, replacement=True)
            actions = []
            for idx in index:
                res[cand_names[idx]] = candidate[cand_names[idx]]
                actions.append(cand_indx[idx])
            # update buffer
            if training:
                self.predAct2.append(actions)
                self.predProb2.append(logits)
            return res
        else:
            # update buffer
            if training:
                self.predAct2.append(cand_indx)
                self.predProb2.append(logits)
            return candidate

    def sample_branching_action(self, state, candidate, training=True):
        # Given action space and current space, sample an action
        cand = state[-1]
        cand_indx = cand[0]
        cand_names = cand[1]
        logits, go_next, before_hand1, before_hand2 = self.RLmodel_branch(self.tensorA, state[0], state[1])
        self.critic_buffer = torch.cat((before_hand1, before_hand2), 1)
        logits_sct = logits[cand_indx]
        index = logits_sct.multinomial(num_samples=1, replacement=True)
        self.go_next_sampled = go_next.multinomial(num_samples=1, replacement=True)
        # update buffer
        if training:
            self.predAct1.append(cand_indx[index])
            self.predProb1.append(logits)
            self.predActDec.append(self.go_next_sampled)
            self.predProbDec.append(go_next)
        return cand_names[index[0]], self.go_next_sampled

    def branch_variable_selection(self, training=True):
        """
            Select a set of variables to branch in RLBC algorithm

            return: self.branchVar, a dictionary with key: variable name, value: variable value
        """
        # according to self.branch_node, choose a self.branch_variable 
        # choose the variable to branch: Maximum Fractionality Rule
        node = self.branch_bound_tree[self.branch_node]
        number_of_noninteger = len(node['fractional_int'])
        number_of_nonbinary = len(node['fractional_bin'])
        if self.branchVariableSelectionMode == 'MFR':
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
        elif self.branchVariableSelectionMode == 'RAND':
            if number_of_noninteger > 0:
                self.branch_variable = random.choice(list(node['fractional_int'].keys()))
            elif number_of_nonbinary > 0:
                self.branch_variable = random.choice(list(node['fractional_bin'].keys()))
            else:
                print(f'node {self.branch_node} has no fractional variables')
                exit()
        elif self.branchVariableSelectionMode == 'RL':
            candidates = node['fractional_int'].copy()
            candidates.update(node['fractional_int'])
            # check if model loaded
            if self.RLmodel_branch is None:
                print('ERROR:: model not loaded')
                quit()
            # query the current state
            state = self.query_state(node, candidates)

            # sample action
            self.branch_variable, self.go_next = self.sample_branching_action(state, candidates, training)

    def naive_cpt_variable_selection(self, node_index, training=True):
        """
            Select a set of variables to branch in naive CPT algorithm

            return: self.branchVar, a dictionary with key: variable name, value: variable value
        """
        # super().branch_variable_selection()  # method overriding: You can call the original method, too, if needed
        number_of_candidates = self.number_branch_var  # the number of variables that are chosen to branch, so the number of nodes in the branching tree is 2^number_of_candidates
        node = self.branch_bound_tree[node_index]
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
                print(f'node {node_index} has no fractional variables')
                exit()
        elif self.cptVariableSelectionMode == 'RL':
            candidates = node['fractional_int'].copy()
            candidates.update(node['fractional_int'])
            # check if model loaded
            if self.RLmodel is None:
                print('ERROR:: model not loaded')
                quit()
            # query the current state
            state = self.query_state(node, candidates)

            # sample action
            self.branchVar = self.sample_disjunction_action(state, candidates, training)

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

        self.bounding_problem.setObjective(0, GRB.MINIMIZE)
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
        right_node['trace'].append([self.branch_variable, '>', math.floor(node['sol'][pos]) + 1])  # x >= ceil(xhat)
        right_node['cutA'] = deepcopy(node['cutA'])
        right_node['cutRHS'] = deepcopy(node['cutRHS'])

        left_node_ind = max(self.branch_bound_tree.keys()) + 1
        right_node_ind = left_node_ind + 1
        self.branch_bound_tree[left_node_ind] = left_node
        self.branch_bound_tree[right_node_ind] = right_node
        feasibilityLeft = self.nodal_problem(left_node_ind)
        feasibilityRight = self.nodal_problem(right_node_ind)
        # feasibilityLeft = self.feasibilityTest(left_node_ind, self.branch_bound_tree)
        # feasibilityRight = self.feasibilityTest(right_node_ind, self.branch_bound_tree)

        # build a naive CPT tree for the left node (Node-Based Branching)
        if self.disjunctiveCut and self.go_next:
            if feasibilityLeft:
                self.cutting_plane_tree = {}
                self.naive_cpt_variable_selection(left_node_ind, training=self.train)
                self.cutting_plane_tree[left_node_ind] = left_node
                self.cutting_plane_tree_building(0, left_node_ind, self.branch_bound_tree[self.branch_node]['sol'])
                # select nodes to generate cuts
                self.cut_generation_node_selection()
                # generate cuts
                self.cut_generation(left_node_ind)
            # build a naive CPT tree for the right node (Node-Based Branching)
            if feasibilityRight:
                self.cutting_plane_tree = {}
                self.naive_cpt_variable_selection(right_node_ind, training=self.train)
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
            node = self.cutting_plane_tree[branch_node_ind]  # father node
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
            right_node['trace'].append([branch_variable, '>', math.floor(sol[pos]) + 1])  # x >= ceil(xhat)

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

        num_constrs, num_vars = self.A.shape[0], self.A.shape[
            1]  # self.mipModel.getAttr('NumConstrs'), self.mipModel.getAttr('NumVars')
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

            num_cuts = len(self.cutting_plane_tree[node_index]['cutRHS']) if self.cutting_plane_tree[node_index][
                                                                                 'cutRHS'] != None else 0
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
                constr_index = self.cutting_plane_tree[node_index]['cutA'].getcol(i).nonzero()[0] if \
                self.cutting_plane_tree[node_index][
                    'cutRHS'] != None else []  # the set of constraints that contain the variable 'var'
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

    def solve(self, eps=None):
        if eps is not None:
            self.load_from_checkpoint(eps)
        time_init = time.time()
        self.train = False
        while self.iteration <= self.maxIteration:
            iter_begin = time.time()
            # selection a node to branch
            if self.iteration == 0:
                self.branch_node_selection()
            if self.OPT == True or self.upper_bound['value'] - self.lower_bound['value'] <= 1e-1:
                self.print_iteration_info()
                return
            # selection a variable to branch
            self.branch_variable_selection(self.train)
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

    def train(self, eps=None):
        if eps is not None:
            self.load_from_checkpoint(eps)
        time_init = time.time()
        self.last_bound = self.lower_bound['value']
        while self.iteration <= self.maxIteration:
            iter_begin = time.time()
            # selection a node to branch
            if self.iteration == 0:
                self.branch_node_selection()
            if self.OPT == True or self.upper_bound['value'] - self.lower_bound['value'] <= 1e-1:
                self.print_iteration_info()
                return
            # selection a variable to branch
            self.branch_variable_selection(self.train)
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
            self.buffer_length += 1
            # print(self.predProb1[-1].shape,self.predProb2[-1].shape,self.predProbDec[-1].shape)
            if len(self.rewards) == 0:
                self.rewards.append((self.lower_bound['value'] - self.last_bound) / self.last_bound - 0.2)
            else:
                self.rewards.append(
                    (self.lower_bound['value'] - self.last_bound - self.rewards[-1]) / self.last_bound - 0.2)
            self.predVals.append(self.critic(self.critic_buffer))
            if self.buffer_length >= self.maxBuffer:
                if self.lock is not None:
                    if self.global_model is not None:
                        with self.lock:
                            self.learnFromBuffer_global()
                            self.last_bound = self.lower_bound['value']
                    else:
                        with self.lock:
                            self.learnFromBuffer()
                            self.last_bound = self.lower_bound['value']
                else:
                    if self.global_model is not None:
                        self.learnFromBuffer_global()
                        self.last_bound = self.lower_bound['value']
                    else:
                        self.learnFromBuffer()
                        self.last_bound = self.lower_bound['value']
