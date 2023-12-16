from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB
import torch
import torch.nn as nn
import numpy as np
import os
import math
import time
from tqdm import tqdm
from alive_progress import alive_bar

from opt_alg.cutting_plane_framework import CuttingPlaneMethod
from rl_alg.model import var_sorter


class NaiveCuttingPlaneTreeAlgorithm(CuttingPlaneMethod):
    def __init__(self, instanceName, maxIteration=100,
                 OutputFlag=0, Threads=1, MIPGap=0.0, TimeLimit=3600, MIPFocus=2, cglp_OutputFlag=0,
                 cglp_Threads=1, cglp_MIPGap=0.0, cglp_TimeLimit=100, cglp_MIPFocus=0,
                 addCutToMIP=False, number_branch_var=2, normalization='SNC',
                 training=True, load_ckp=True, save_interval=20
                 ):
        super().__init__(instanceName, maxIteration, OutputFlag, Threads, MIPGap, TimeLimit, MIPFocus, cglp_OutputFlag,
                         cglp_Threads, cglp_MIPGap, cglp_TimeLimit, cglp_MIPFocus, addCutToMIP, number_branch_var,
                         normalization)
        self.tensorA = None
        self.col_feature = None
        self.row_feature = None
        # model
        self.pred = var_sorter(v_size=6, c_size=3, sample_sizes=[64], multi_head=2, natt=2)
        # optimizer
        self.optimizer = torch.optim.Adam(self.pred.parameters(), lr=5e-5)
        # training params
        self.training = training
        self.lastLogit = []
        self.save_interval = save_interval
        if not os.path.isdir('./models'):  # create model folder if there is no models folder
            os.mkdir('./models')
        self.ckp_starter = 0
        if load_ckp:  # checkpoint: load model and optimizer; start from the latest checkpoint
            '''get latest checkpoint'''
            # resume model training from the latest checkpoint, allowing the training to continue from the state it
            # was in when interrupted, rather than starting over from the beginning.
            ckps = os.listdir('./models')  # get all ckps (files with .mdl extension) in this directory
            ckps = [f for f in ckps if f.endswith('.mdl')]
            if len(ckps) > 0:
                ckps.sort(key=lambda x: int(x.replace('.mdl', '').split('_')[-1]))
                tar_name = f'./models/{ckps[-1]}'
                # load
                cpu_dev = torch.device('cpu')
                checkpoint = torch.load(tar_name, map_location=cpu_dev)  # load model and optimizer
                self.pred.load_state_dict(checkpoint['model'])  # load model
                self.ckp_starter = checkpoint['nepoch']
                print(f'Loaded check point: {tar_name}')
            else:
                print('No check point to load')
        # set up log files
        if not os.path.isdir('./logs'):
            os.mkdir('./logs')
        self.log_file = open('./logs/training.log', 'w')  # 'w' means: if the file exists, it will be overwritten

    def A_to_sparse_tensor(self):
        self.tensorA = self.A.tocoo()
        indices = np.vstack((self.tensorA.row, self.tensorA.col))
        data = self.tensorA.data
        self.tensorA = torch.sparse_coo_tensor(indices, data, self.tensorA.shape, dtype=torch.float32)

    def feat_extract(self):
        # A matrix update
        self.A_to_sparse_tensor()

        # variable feature update
        col_feat = [None] * len(self.variables)
        for var in self.variables:
            rel_sol_indx = self.varName_map_position[var.VarName]
            col_value = self.lp_sol[rel_sol_indx]
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
        cand = torch.as_tensor([self.varName_map_position[x] for x in self.Cand])
        return col_feat, row_feat, row_index_map, cand

    def getPred(self):
        number_of_candidates = self.number_branch_var
        # get feature, candidates are stored in self.Cand
        col_feat, row_feat, row_index_map, cand = self.feat_extract()

        # calling model to do prediction
        self.optimizer.zero_grad()
        self.lastLogit.append(self.pred(self.tensorA, col_feat, row_feat))
        decision = torch.argsort(torch.index_select(self.lastLogit[-1], 0, cand), descending=True)[
                   :number_of_candidates]
        self.branchVar[self.iteration - 1] = {}
        for i in decision:
            tar_var = self.Cand[i]
            # ori_ind = cand[i].item()
            self.branchVar[self.iteration - 1][tar_var] = self.lastLogit[-1][i]  # self.lp_sol[ori_ind]

    def variable_selection(self, ifTrain=True, variable_selection_way='MFR'):
        # choose the variable to branch
        # TODO: add criteria to select either explore or exploit
        # when we will do heuristic branching, when we will explore, and when we will exploit?
        if variable_selection_way == 'RL':
            # the number of variables that are chosen to branch, so the number of nodes in the branching tree is
            # 2^number_of_candidates
            number_of_candidates = self.number_branch_var
            # get feature, candidates are stored in self.Cand
            col_feat, row_feat, row_index_map, cand = self.feat_extract()

            # calling model to do prediction
            if ifTrain:
                self.optimizer.zero_grad()
            self.lastLogit.append(self.pred(self.tensorA, col_feat, row_feat))
            decision = torch.argsort(torch.index_select(self.lastLogit[-1], 0, cand), descending=True)[
                       :number_of_candidates]
            self.branchVar[self.iteration - 1] = {}
            for i in decision:
                tar_var = self.Cand[i]
                # ori_ind = cand[i].item()
                self.branchVar[self.iteration - 1][tar_var] = self.lastLogit[-1][i]  # self.lp_sol[ori_ind]
        elif variable_selection_way == 'MFR':
            if ifTrain:
                self.getPred()
            # heuristic: Maximum Fractionality Rule
            number_of_candidates = self.number_branch_var  # the number of variables that are chosen to branch, so the number of nodes in the branching tree is 2^number_of_candidates
            number_of_noninteger = len(self.non_integer_vars[self.iteration - 1])
            number_of_nonbinary = len(self.non_binary_vars[self.iteration - 1])
            self.branchVar[self.iteration - 1] = {}
            if number_of_noninteger > 0:
                list1 = sorted(self.non_integer_vars[self.iteration - 1].items(), key=lambda x: x[1], reverse=True)[
                        :number_of_candidates]  # find the integer variables that have the largest distance to the nearest integer
                if len(list1) <= number_of_candidates:
                    for item in list1:
                        self.branchVar[self.iteration - 1][item[0]] = item[1]  #
                else:
                    for item in list1[0:number_of_candidates]:
                        self.branchVar[self.iteration - 1][item[0]] = item[1]

            if number_of_nonbinary > 0:
                list2 = sorted(self.non_binary_vars[self.iteration - 1].items(), key=lambda x: x[1], reverse=True)[
                        :number_of_candidates]  # find the binary variables that have the largest distance to {0,1}
                if len(list2) <= number_of_candidates:
                    for item in list2:
                        self.branchVar[self.iteration - 1][item[0]] = item[1]
                else:
                    for item in list2[0:number_of_candidates]:
                        self.branchVar[self.iteration - 1][item[0]] = item[1]

    def branching_tree_building(self, node, level, varInfo):
        if level == len(self.branchVar[self.iteration - 1]):
            return
        else:
            varName, info = list(varInfo.items())[level]
            pos = self.varName_map_position[varName]

            left_node = {}
            left_node['LB'] = deepcopy(self.nodeSet[node]['LB'])
            left_node['LB'][pos] = info['upper']
            left_node['UB'] = deepcopy(self.nodeSet[node]['UB'])
            left_node['trace'] = deepcopy(self.nodeSet[node]['trace'])
            left_node['trace'].append('l')

            right_node = {}
            right_node['LB'] = deepcopy(self.nodeSet[node]['LB'])
            right_node['UB'] = deepcopy(self.nodeSet[node]['UB'])
            right_node['UB'][pos] = info['lower']
            right_node['trace'] = deepcopy(self.nodeSet[node]['trace'])
            right_node['trace'].append('r')

            left_node_ind = max(self.nodeSet.keys()) + 1
            right_node_ind = left_node_ind + 1
            self.nodeSet[left_node_ind] = left_node
            self.nodeSet[right_node_ind] = right_node
            del self.nodeSet[node]

            self.branching_tree_building(left_node_ind, level + 1, varInfo)
            self.branching_tree_building(right_node_ind, level + 1, varInfo)

    def branching_tree(self):
        varInfo = {}
        for varName in self.branchVar[self.iteration - 1].keys():
            varInfo[varName] = {}
            varInfo[varName]['val'] = self.lp_relaxation.getVarByName(varName).x
            varInfo[varName]['lower'] = math.floor(varInfo[varName]['val'])
            varInfo[varName]['upper'] = math.ceil(varInfo[varName]['val'])

        self.nodeSet = {}
        self.nodeSet[0] = {}
        self.nodeSet[0]['LB'] = deepcopy(self.LB)
        self.nodeSet[0]['UB'] = deepcopy(self.UB)
        self.nodeSet[0]['trace'] = []
        self.branching_tree_building(0, 0, varInfo)

    def solve(self, ifTrain=False, variable_selection_way='RL'):
        time_init = time.time()
        f = open(f'./logs/eval_train_{variable_selection_way}.log', 'w')
        while self.iteration <= self.maxIteration:
            iter_begin = time.time()
            self.master_problem()
            if self.OPT == True:
                self.print_iteration_info()
                return
            self.variable_selection(ifTrain, variable_selection_way)
            self.branching_tree()
            ready_to_cut = time.time()
            self.cut_generation()
            iter_end = time.time()
            overall = iter_end - time_init
            iteration_time = iter_end - iter_begin
            cut_time = iter_end - ready_to_cut
            # print(self.lp_obj_value)
            f.write(f'iter:{self.iteration} lpobj:{self.lp_obj_value[self.iteration - 1]}\n')
            self.print_iteration_info(cut_time, iteration_time, overall)
        f.close()

    def train_model_each_iteration(self):
        time_init = time.time()
        while self.iteration <= self.maxIteration:
            iter_begin = time.time()
            self.master_problem()
            if self.OPT == True:
                self.print_iteration_info()
                return
            self.variable_selection(True, 'RL')
            self.branching_tree()
            ready_to_cut = time.time()
            self.cut_generation()
            iter_end = time.time()
            overall = iter_end - time_init
            iteration_time = iter_end - iter_begin
            cut_time = iter_end - ready_to_cut
            self.print_iteration_info(cut_time, iteration_time, overall)
            if self.iteration > 1 and self.training:
                # compute improvement
                improvement = (self.lp_obj_value[self.iteration - 1] - self.lp_obj_value[self.iteration - 2]) / \
                              self.lp_obj_value[0] * 100
                # record reward
                self.log_file.write(f'iter:{self.iteration} reward:{improvement}\n')
                # update gradient and update model
                regret = torch.sum(self.lastLogit[-1] * ((-1.0) * improvement + 1e-6))
                regret.backward()
                self.optimizer.step()
                # check if need to save model
                if (self.iteration - 1) % self.save_interval == 0 and self.iteration - 1 != 0:
                    state = {'model': self.pred.state_dict(), 'optimizer': self.optimizer.state_dict(),
                             'nepoch': self.ckp_starter + self.iteration - 1}
                    torch.save(state, f'./models/ckp_{self.iteration + self.ckp_starter - 1}.mdl')

    def train_model_each_round(self, num_episodes=500):
        # TODO: train model after each round, not each iteration; and for each round, we need to initialize the model (cuts, variables, etc.):
        '''         # Initialize the model:
                    self.iteration = 0
                    self.lp_relaxation = self.mipModel.relax()
                    self.lp_relaxation.update()
                    self.coefList = {}
                    self.lp_obj_value = {}
        '''
        for i in range(10):
            # with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            with alive_bar(int(num_episodes / 10), title=f'Iteration {i}', bar='bubbles', spinner='arrow') as pbar:
                for i_episode in range(int(num_episodes / 10)):
                    t_imp = 0
                    regret = None
                    time_init = time.time()
                    self.iteration = 0
                    self.lp_relaxation = self.mipModel.relax()
                    self.lp_relaxation.update()
                    self.coefList = {}
                    self.lp_obj_value = {}
                    while self.iteration <= self.maxIteration:
                        iter_begin = time.time()
                        self.master_problem()
                        self.variable_selection(variable_selection_way='RL')
                        self.branching_tree()
                        ready_to_cut = time.time()
                        self.cut_generation()
                        iter_end = time.time()
                        overall = iter_end - time_init
                        iteration_time = iter_end - iter_begin
                        cut_time = iter_end - ready_to_cut
                        self.print_iteration_info(cut_time, iteration_time, overall)
                        ind = [self.varName_map_position[x] for x in self.Cand]

                        # compute improvement
                        improvement = abs(self.lp_obj_value[self.iteration - 1] - self.lp_obj_value[0]) / \
                                      self.lp_obj_value[0] * 100
                        t_imp += improvement
                        # record reward
                        self.log_file.write(f'episode:{i * 10 + i_episode + 1} reward:{improvement}\n')
                        # update gradient and update model
                        if regret is None:
                            regret = torch.sum(self.lastLogit[-1][ind] * ((-1.0) * improvement + 0.1))
                        else:
                            regret += torch.sum(self.lastLogit[-1][ind] * ((-1.0) * improvement + 0.1))
                    regret.backward()
                    self.optimizer.step()
                    print('***** Gradient Applied *****')
                    # check if need to save model
                    state = {'model': self.pred.state_dict(), 'optimizer': self.optimizer.state_dict(),
                             'nepoch': self.ckp_starter + i_episode + int(num_episodes / 10) * i}
                    torch.save(state, f'./models/ckp_{i_episode + self.ckp_starter + int(num_episodes / 10) * i}.mdl')
                    pbar(1)
                    self.log_file.write(f'average improvement:{t_imp / self.maxIteration}\n')

    def train_model_each_round_warmStart(self, num_episodes=500):
        # TODO: train model after each round, not each iteration; and for each round, we need to initialize the model (cuts, variables, etc.):
        '''         # Initialize the model:
                    self.iteration = 0
                    self.lp_relaxation = self.mipModel.relax()
                    self.lp_relaxation.update()
                    self.coefList = {}
                    self.lp_obj_value = {}
        '''
        # with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        with alive_bar(int(num_episodes), title=f'Iteration WarmStart', bar='bubbles', spinner='arrow') as pbar:
            for i_episode in range(int(num_episodes)):
                regret = None
                time_init = time.time()
                self.iteration = 0
                self.lp_relaxation = self.mipModel.relax()
                self.lp_relaxation.update()
                self.coefList = {}
                self.lp_obj_value = {}
                while self.iteration <= self.maxIteration:
                    iter_begin = time.time()
                    self.master_problem()
                    self.variable_selection(variable_selection_way='MFR')
                    self.branching_tree()
                    ready_to_cut = time.time()
                    self.cut_generation()
                    iter_end = time.time()
                    overall = iter_end - time_init
                    iteration_time = iter_end - iter_begin
                    cut_time = iter_end - ready_to_cut
                    self.print_iteration_info(cut_time, iteration_time, overall)
                    # compute improvement
                    improvement = abs(self.lp_obj_value[self.iteration - 1] - self.lp_obj_value[0]) / self.lp_obj_value[
                        0] * 100
                    # record reward
                    self.log_file.write(f'Warm Start----episode:{self.iteration} reward:{improvement}\n')
                    # update gradient and update model
                    if regret is None:
                        tz = torch.zeros(self.lastLogit[-1].shape)
                        indc = [self.varName_map_position[x] for x in self.branchVar[self.iteration - 1]]
                        tz[indc] = 1.0
                        regret = torch.nn.functional.cross_entropy(self.lastLogit[-1], tz)
                    else:
                        tz = torch.zeros(self.lastLogit[-1].shape)
                        indc = [self.varName_map_position[x] for x in self.branchVar[self.iteration - 1]]
                        tz[indc] = 1.0
                        regret = torch.nn.functional.cross_entropy(self.lastLogit[-1], tz)
                regret.backward()
                self.optimizer.step()
                print('***** Gradient Applied *****')
                # check if need to save model
                # state = {'model':self.pred.state_dict(),'optimizer':self.optimizer.state_dict(),'nepoch':self.ckp_starter+i_episode+int(num_episodes / 10)*i}
                # torch.save(state,f'./models/ckp_{i_episode+self.ckp_starter+int(num_episodes / 10)*i}.mdl')
                pbar(1)
