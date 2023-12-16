import gurobipy as gp
import torch
import torch.nn as nn
import os
from code_LY.helper import getCandidates
from code_LY.model import var_sorter

# torch devices
# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
cpu_dev = torch.device('cpu')

# adding args
OUTPUT = True
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--problem', type=str, default='ip')
parser.add_argument('-n', '--nnode', type=int, default=5)
parser.add_argument('-s', '--nstep', type=int, default=100)
parser.add_argument('-l', '--lr', type=float, default=1e-3)
args = parser.parse_args()

# set up torch model
m = var_sorter(6, 2, [64, 128])

# load instance info
ins_dir = f'/Users/aaron/Downloads/benchmark/50v-10.mps.gz'
# mscp=gp.Model()
mscp1 = gp.read(ins_dir)

# construct CGLP
mcglp = gp.Model()
# mcglp.enableReoptimization()
cglp_var = []
cglp_mult = []
cglp_coeff = []

# get variables global
mvars = mscp1.getVars()
vmap = {}
v_indx_map = {}
v_list = []
cglp_var.append(mcglp.addVar(vtype=gp.GRB.CONTINUOUS, name=f'pi_0', obj=-1.0))
# cglp_coeff.append(-1.0)
vct = 0
for v in mvars:
    vnm = v.getAttr('VarName')
    vmap[vnm] = v
    v_indx_map[vnm] = vct
    v_list.append(v)
    vct += 1
    # CGLP var
    tmp_var = mcglp.addVar(vtype=gp.GRB.CONTINUOUS, name=f'pi_{vnm}', obj=0.0)
    cglp_var.append(tmp_var)
    # cglp_coeff.append(0.0)
mcglp.update()
# current_nrow=mscp.getNConss()
current_nrow = mscp1.getAttr('NumConstrs')
print(f'nCons: {current_nrow}')

training = True

last_obj = -1e+20
node_limit = 0

step = 0
init_obj = None

optimizer = torch.optim.Adam(m.parameters(), lr=args.lr)
loss_func = torch.nn.MSELoss()

# for step in range(args.nstep):
# node_limit+=1
# mscp.setParam('NodeLimit',1)
# mscp.writeProblem('verifyMSCP.lp')

# start training cycle

labels = None
logits = None


def cbk(mscp, where):
    global step
    global last_obj
    global init_obj
    global optimizer
    global loss_func
    global training
    global labels
    global logits
    if not where == gp.GRB.Callback.MIPNODE:
        return
    # step=mscp.cbGet(gp.GRB.Callback.MIPNODE_NODCNT)
    step += 1
    print(f'calling callback step{step}')
    current_nrow = mscp.getAttr('NumConstrs')

    # get A matrix
    Ak = []  # sparse matrix, indexed by 0..m-1, inside stored as tuple (0..n-1,val)
    rhs = [0] * current_nrow
    for i in range(len(vmap)):
        Ak.append([])
    for cidx, c in enumerate(mscp.getConstrs()):
        row = mscp.getRow(c)
        row_nv = row.size()
        rhs[cidx] = c.RHS
        for col_indx in range(row_nv):
            vnm2 = row.getVar(col_indx).VarName
            vidx = v_indx_map[vnm2]
            Ak[vidx].append((cidx, row.getCoeff(col_indx)))
            # print(vnm,cidx,vidx,coeffs[vnm])
    # quit()

    # get current node embedding
    cand, col_feat, row_feat, A, col_index_map, lp_sol_map = getCandidates(mscp, vmap)
    lpobj = mscp.cbGet(gp.GRB.Callback.MIPNODE_OBJBND)
    # print(col_feat)
    # quit()
    # update records
    improvement = 0.0
    if last_obj < lpobj:
        improvement = lpobj - last_obj

    # TODO::add training
    if training and init_obj is not None:
        label = torch.zeros(logits.shape)
        label[labels] = 1.0 * 1000.0 * (improvement / init_obj)
        # print(label.shape,labels,label)
        # quit()
        loss = loss_func(logits, label)
        loss.backward()
        optimizer.step()
    elif init_obj is None:
        init_obj = lpobj
    # if training:

    # if step==0:
    #     last_obj=lpobj
    #     print(f'Initial Bound/LP Obj: {last_obj}')
    #     continue

    # predict
    if training:
        optimizer.zero_grad()
    print('----------Predicting.....', end='')
    logits = m(A, col_feat, row_feat)
    print('Done---------')
    cand_indx = [col_index_map[x] for x in cand]

    logits_map = {}
    for idx, vname in enumerate(cand):
        logits_map[vname] = [logits[cand_indx[idx]].item(), idx]

    logits_map = sorted(logits_map.items(), key=lambda x: x[1][0], reverse=True)[:args.nnode]

    # for debuge
    predicted_ = set()
    labels = []
    if OUTPUT:
        print(
            f'  ::::::  Step: {step}, #rows:{current_nrow}\n        --- LP Obj  : {lpobj}\n        --- last Obj: {last_obj}\n        *** imp: {improvement}')
        print(f'---------------------------------------')
        print(f'|  #  |  VName  |  logits  |  LP Sol  |')
        print(f'---------------------------------------')
        # '{:10s} {:3d}  {:7.2f}'.format('xxx', 123, 98)
        for idx, ele in enumerate(logits_map):
            print('| ' + '{:4d}'.format(idx) + '|  ' + '{:7s}'.format(ele[0]) + '| ' + '{:8.4f}'.format(
                ele[1][0]) + ' | ' + '{:8.4f}'.format(lp_sol_map[ele[0]]) + ' |')
            predicted_.add(ele[0])
            labels.append(ele[1][1])
        print(f'---------------------------------------')
        input('Press key to continue')

    last_obj = lpobj

    # Update CGLP
    for cglp_idx in range(1, len(cglp_var)):
        v = cglp_var[cglp_idx]
        v.setAttr('Obj', lp_sol_map[v.VarName.replace('pi_', '')])
    # add constraints
    # 1. pi constraints (m)
    tmp_lambda = []
    tmp_mu = []
    tmp_v = []
    for i in range(0, len(cglp_var) - 1):
        tmp_var = mcglp.addVar(vtype=gp.GRB.CONTINUOUS, name=f'mu_{step}_{i}', obj=0.0)
        tmp_mu.append(tmp_var)
        tmp_var = mcglp.addVar(vtype=gp.GRB.CONTINUOUS, name=f'v_{step}_{i}', obj=0.0)
        tmp_v.append(tmp_var)
    for i in range(current_nrow):
        tmp_var = mcglp.addVar(vtype=gp.GRB.CONTINUOUS, name=f'lambda_{step}_{i}', obj=0.0)
        tmp_lambda.append(tmp_var)
    for i in range(1, len(cglp_var)):
        tmp_coeff = []
        # var,coeff
        tmp_coeff.append((cglp_var[i], -1))

        # deal with mu/v
        tmp_bound = v_list[i - 1].LB
        if tmp_bound > -1e+20:
            tmp_coeff.append((tmp_mu[i - 1], 1))
        tmp_bound = v_list[i - 1].UB
        if tmp_bound < 1e+20:
            tmp_coeff.append((tmp_v[i - 1], -1))
            # deal with lambda
        #  first with A, then with cuts
        #   basically for pi_m, we need to compute A[:,i-1]^\top \lambda
        for nindx, ent in enumerate(Ak[i - 1]):
            tmp_coeff.append((tmp_lambda[ent[0]], ent[-1]))
        mcglp.addConstr(gp.quicksum(k[0] * k[1] for k in tmp_coeff) == 0, f'cons_{step}_pi_{i}')
    tmp_coeff = [(cglp_var[0], -1)]
    for i in range(current_nrow):
        tmp_coeff.append((tmp_lambda[i], rhs[i]))
    for i in range(1, len(cglp_var)):
        vnm = v_list[i - 1].VarName
        tmp_bound = round(mscp.cbGetNodeRel(v_list[i - 1]))

        if vnm not in predicted_:
            tmp_bound = v_list[i - 1].LB
        if tmp_bound <= -1e+20:
            tmp_bound = 0
        else:
            tmp_coeff.append((tmp_mu[i - 1], tmp_bound))
        if vnm not in predicted_:
            tmp_bound = v_list[i - 1].UB
        if tmp_bound >= 1e+20:
            tmp_bound = 0
        else:
            tmp_coeff.append((tmp_v[i - 1], tmp_bound))
    mcglp.addConstr(gp.quicksum(k[0] * k[1] for k in tmp_coeff) >= 0, f'cons_{step}_pi0')
    mcglp.addConstr(gp.quicksum(tmp_lambda + tmp_mu + tmp_v) == 1, f'norm_{step}')

    mcglp.write(f'verify{step}.lp')

    mcglp.update()
    mcglp.optimize()
    vss = mcglp.getVars()
    # for v in vss:
    #    print(v.name,mcglp.getVal(v))

    local_vmap = {}
    for v in mscp.getVars():
        local_vmap[v.VarName] = v

    newCut = gp.LinExpr(cglp_var[0].X)
    for v in cglp_var[1:]:
        ori_name = v.VarName[3:]
        ori_var = local_vmap[ori_name]
        # print(v,ori_name,v.X,ori_var)
        newCut += ori_var * v.X
    mscp.cbCut(newCut >= 0.0)
    print(f'Finished step{step}')


mscp1.optimize(cbk)
