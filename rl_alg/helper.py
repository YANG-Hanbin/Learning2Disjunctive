# import pyscipopt as scp
import torch
import torch.nn as nn
import os


#constants
LP_SOLVED_OPTIMAL=1

# get candidates based on the model's current LP
def getCandidates(mscp,vmap):
    cand=[]
    col_feat=[]
    row_feat=[]
    edge_ind=[[],[]]
    edge_val=[]
    lp_sol_map={}
    ccounter=0
    rcounter=0
    
        
    #setup helpers
    col_index_map={}
    row_index_map={}
    
    # dealing with columns
    mvs=mscp.getVars()
    rel_sol=mscp.cbGetNodeRel(mvs)   # Gets the relaxed solution of variables at the current node in a MIP model.

    sol_map={}
    for rel_sol_indx,col_var in enumerate(mvs):    
        col_val=rel_sol[rel_sol_indx]
        sol_map[col_var.VarName]=col_val
        vtp=vmap[col_var.VarName].VType
        # print(col_var.VarName,col_val,vtp)
        if abs(col_val-round(col_val,0))>1e-6 and (vtp=='B' or vtp=='I'):
            cand.append(col_var.VarName)
        #compose features
        isBin=0
        isInt=0
        if vtp=='I':
            isInt=1
        elif vtp=='B':
            isBin=1
        # rc=col_var.RC
        # ---- [lp_sol, LB, UB, isBin, isInt, reduced_cost, lp_obj]
        tmp=[col_val,col_var.LB,min(col_var.UB,1e+20), isBin, isInt, col_var.Obj]    
        lp_sol_map[col_var.VarName] = col_val
        if col_var.VarName not in col_index_map:
            col_index_map[col_var.VarName]=ccounter
            ccounter+=1
        #print(col_var.name,tmp)
        #add to return
        col_feat.append(tmp)
    print(f'total vars:{len(mvs)}, frac/cand vars:{len(cand)}')

    #dealing with rows
    for cidx,c in enumerate(mscp.getConstrs()):


        if c.ConstrName not in row_index_map:
            row_index_map[c.ConstrName]=rcounter
            rcounter+=1
        row_ind=row_index_map[c.ConstrName]
        #compose row feature
        rhs=c.RHS
        #norm=row.getNorm()  #----not supported on my device
        sense=0 # 0:leq 0.5:eq 1:geq
        if c.Sense=='=':
            sense=0.5
        elif c.Sense=='>':
            sense=1
        # slack=c.Slack
        # dual=c.Pi
        row=mscp.getRow(c)
        # lhs=row.getValue()
        row_feat.append([rhs,sense])



        row_nv=row.size()
        for col_indx in range(row_nv):
            col_var=row.getVar(col_indx)
            vnm2=col_var.VarName
            col_ind=col_index_map[vnm2]
            edge_ind[0].append(row_ind)
            edge_ind[1].append(col_ind)
            edge_val.append(row.getCoeff(col_indx))


    A=torch.sparse_coo_tensor(edge_ind, edge_val, (len(row_feat),len(col_feat)),dtype=torch.float32)
    col_feat=torch.as_tensor(col_feat,dtype=torch.float32)
    row_feat=torch.as_tensor(row_feat,dtype=torch.float32)
    return cand,col_feat, row_feat, A, col_index_map, lp_sol_map
    













def getCandidates_betsuMethod(mscp,custom_extract):
    cand=[]
    col_feat=[]
    row_feat=[]
    edge_ind=[[],[]]
    edge_val=[]
    lp_sol_map={}
    ccounter=0
    rcounter=0
    if mscp.getLPSolstat()==LP_SOLVED_OPTIMAL:
        print(f'lp stat: {LP_SOLVED_OPTIMAL}')
        
        #setup helpers
        col_index_map={}
        row_index_map={}
        
        # dealing with columns
        current_lp_cols=mscp.getLPColsData()
        cand=custom_extract(mscp,current_lp_cols)
        for col in current_lp_cols:
            col_var=col.getVar()
            col_val=col.getPrimsol()
            #compose features
            isBin=0
            isInt=0
            if col_var.vtype()=='INTEGER':
                isInt=1
            elif col_var.vtype()=='BINARY':
                isBin=1
            rc=mscp.getVarRedcost(col_var)
            # ---- [lp_sol, LB, UB, isBin, isInt, reduced_cost, lp_obj]
            tmp=[col_val,col.getLb(),col.getUb(), isBin, isInt, rc, col_var.getObj()]    
            lp_sol_map[col_var.name.replace('t_','')] = col_val
            if col_var.name not in col_index_map:
                col_index_map[col_var.name]=ccounter
                ccounter+=1
            #print(col_var.name,tmp)
            #add to return
            col_feat.append(tmp)
        print(f'total vars:{len(current_lp_cols)}, frac/cand vars:{len(cand)}')
        
        #dealing with rows
        current_lp_rows=mscp.getLPRowsData()
        for row in current_lp_rows: 
            if row.name not in row_index_map:
                row_index_map[row.name]=rcounter
                rcounter+=1
            row_ind=row_index_map[row.name]
            #compose row feature
            rhs=row.getRhs()
            lhs=row.getLhs()
            #norm=row.getNorm()  #----not supported on my device
            sense=0 # 0:leq 0.5:eq 1:geq
            if lhs==rhs:
                sense=0.5
            elif rhs>=1e+20:
                sense=1
            constant=row.getConstant()
            #print(row.name,lhs-constant,rhs-constant,sense)
            row_feat.append([lhs-constant,rhs-constant,sense,constant])
            row_cols=row.getCols()
            row_coeff=row.getVals()
            
            for order_idx,c in enumerate(row_cols):
                col_var=c.getVar()
                col_ind=col_index_map[col_var.name]
                edge_ind[0].append(row_ind)
                edge_ind[1].append(col_ind)
                edge_val.append(row_coeff[order_idx])
                #print(col_var.name,col_ind,row_coeff[order_idx])
            #input()
        A=torch.sparse_coo_tensor(edge_ind, edge_val, (len(row_feat),len(col_feat)),dtype=torch.float32)
        col_feat=torch.as_tensor(col_feat,dtype=torch.float32)
        row_feat=torch.as_tensor(row_feat,dtype=torch.float32)
        return cand,col_feat, row_feat, A, col_index_map, lp_sol_map, mscp.getLPObjVal()
    return None
    
    



