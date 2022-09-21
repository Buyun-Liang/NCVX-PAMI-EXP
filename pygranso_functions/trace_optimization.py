import torch
from pygranso.pygransoStruct import pygransoStruct

def user_fn(X_struct,A,d,device,folding_type):
    assert folding_type in [None, "l2"], "unidentified folding type for PyGranso."
    V = X_struct.V
    # objective function
    f = -torch.trace(V.T@A@V)
    # inequality constraint, matrix form
    ci = None
    # equality constraint
    ce = pygransoStruct()
    constr_vec = (V.T@V - torch.eye(d).to(device=device, dtype=torch.double)).reshape(d**2,1)
    # if folding_type == 'l1':
    #     ce.c1 = torch.sum(torch.abs(constr_vec))
    # elif folding_type == 'l2':
    #     ce.c1 = torch.sum(constr_vec**2)**0.5
    # elif folding_type == 'linf':
    #     ce.c1 = torch.amax(torch.abs(constr_vec))
    if folding_type == 'l2':
        ce.c1 = torch.sum(constr_vec**2)**0.5
    else: # unfolded
        ce.c1 = V.T@V - torch.eye(d).to(device=device, dtype=torch.double)
        
    return [f,ci,ce]

def opts_init(device,pygranso_config,final_obj_list,data_idx,n,d):
    opts = pygransoStruct()
    opts.torch_device = device
    opts.print_frequency = 10
    opts.maxit = pygranso_config["maxit"]
    opts.print_use_orange = False
    opts.print_ascii = True
    opts.quadprog_info_msg  = False
    opts.opt_tol = pygranso_config["opt_tol"]
    opts.maxclocktime = pygranso_config["maxclocktime"]
    opts.mu0 = pygranso_config["mu0"]
    opts.fvalquit = final_obj_list[data_idx]*pygranso_config["threshold"]
    opts.x0 =  torch.randn((n*d,1)).to(device=device, dtype=torch.double)
    opts.x0 = opts.x0/torch.linalg.norm(opts.x0)
    return opts