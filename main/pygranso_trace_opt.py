# import torch
# import utils
# import sys
# import os
# import numpy as np
# ## Adding PyGRANSO directories. Should be modified by user
# sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
# from pygranso.pygranso import pygranso
# import time
# from matplotlib import pyplot as plt 


import argparse,os, sys, torch, time
import pygranso
sys.path.append("/home/buyun/Documents/GitHub/NCVX-PAMI-EXP")
sys.path.append("/home/jusun/liang664/NCVX-PAMI-EXP")

from utils.config_log_setup import clear_terminal_output, makedir, \
    create_log_info, save_exp_info, set_default_device
from utils.general import load_json, print_and_log
from pygranso_functions.trace_optimization import user_fn, opts_init



# #generate a list of markers and another of colors 
# markers = [ "," , "o" , "v" , "^" , "<", ">", "." ]
# colors = ['r','g','b','c','m', 'y', 'k']

###############################################
# debug_mode = False


def problem_init(n, d, device):
    # data initialization
    A = torch.randn(n,n)
    A = (A + A.T)/2
    # All the user-provided data (vector/matrix/tensor) must be in torch tensor format.
    # As PyTorch tensor is single precision by default, one must explicitly set `dtype=torch.double`.
    # Also, please make sure the device of provided torch tensor is the same as opts.torch_device.
    A = A.to(device=device, dtype=torch.double)
    L, U = torch.linalg.eig(A)
    L = L.to(dtype=torch.double)
    U = U.to(dtype=torch.double)
    index = torch.argsort(L,descending=True)
    U = U[:,index[0:d]]
    analytical_sol = -torch.trace(U.T@A@U).item()
    return [A, U, analytical_sol]

if __name__ == "__main__":
    clear_terminal_output()
    print("Run experiments for trace optimization")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", dest="config", type=str,
        default=os.path.join('config_file_examples', 'trace_optimization.json'),
        help="Path to the json config file."
    )

    args = parser.parse_args()
    cfg = load_json(args.config)  # Load Experiment Configuration file
    
    pygranso_dtype = torch.double # Always use double precision in PyGRANSO
    device, _ = set_default_device(cfg) # Set default device

    # Experiment Root
    save_root = "log_folder" #os.path.join("..", "log_folder")
    root_name = cfg["log_folder"]["save_root"]
    save_root = os.path.join(save_root, root_name)
    makedir(save_root)

    # Experiment ID
    n = cfg["optimization_problem_settings"]["n"] # opt var V: n*d
    d = cfg["optimization_problem_settings"]["d"] # constraint: d*d
    data_matrices_num = cfg["optimization_problem_settings"]["data_matrices_num"] # N different data matrices
    folding_type = cfg["pygranso_options"]["folding_type"]
    restart_num = cfg["pygranso_options"]["restart_num"]
    maxclocktime = cfg["pygranso_options"]["maxclocktime"]

    exp_name = "tr-opt-n%d-d%d-%s-restart%d-time%d" % (
        n, d, folding_type, restart_num, maxclocktime
    )

    check_point_dir = os.path.join(
        save_root, 
        exp_name
    )

    cfg["checkpoint_dir"] = check_point_dir
    makedir(check_point_dir)

    # Create Experiment Log File and save settings
    log_file = create_log_info(check_point_dir)
    save_exp_info(check_point_dir, cfg)
    
    # Create save csv dir
    pygranso_result_csv_dir = os.path.join(
        check_point_dir, "result_summary.csv"
    )

    pygranso_result_summary = {
        "data_idx": [],
        "restart_idx": [],
        "time": [],
        "mean_error": [],
        "F": [],
        "tv": [],
        "MF": [],
        "MF_tv": [],
        "term_code": [],
        "iter": []
    }


    # === Setup Optimization Problem ===
    data_matrices = []
    solution_matrices = []
    final_obj_list = []
    for data_idx in range(data_matrices_num):
        [A, U, analytical_sol] = problem_init(n, d, device)
        data_matrices.append(A)
        solution_matrices.append(U)
        final_obj_list.append(analytical_sol)

    msg = " >> Created %d different trace optimization problems with size n=%d, d=%d."%(data_matrices_num,n,d)
    print_and_log(msg, log_file, mode="w")

    # main function for pygranso
    var_in = {"V": [n,d]}
    pygranso_config = cfg["pygranso_options"]
    
    for data_idx in range(data_matrices_num):
        comb_fn = lambda X_struct : user_fn(X_struct,A,d,device,folding_type)
        opts = opts_init(device,pygranso_config,final_obj_list,data_idx,n,d)
        U_cur = solution_matrices[data_idx]
        for restart_idx in range(restart_num):
            msg = " [%d/%d] restarts. [%d/%d] data matrices. folding type: {} ".format(restart_idx,restart_num,data_idx,data_matrices_num,folding_type)
            print_and_log(msg, log_file, mode="w")
            try:
                # call pygranso
                start = time.time()
                soln = pygranso(var_spec = var_in,combined_fn = comb_fn,user_opts = opts)
                end = time.time()
                pygranso_result_summary["data_idx"].append(data_idx)
                pygranso_result_summary["restart_idx"].append(restart_idx)
                pygranso_result_summary["time"].append(end-start)

                V = torch.reshape(soln.final.x,(n,d)) # reshape final solution
                E = torch.linalg.norm(V-U_cur)/torch.linalg.norm(U_cur) # calculate mean error E
                pygranso_result_summary["mean_error"].append(E)
                pygranso_result_summary["F"].append(soln.final.f)
                pygranso_result_summary["tv"].append(soln.final.tv)
                pygranso_result_summary["MF"].append(soln.most_feasible.f)
                pygranso_result_summary["MF_tv"].append(soln.most_feasible.tv)
                pygranso_result_summary["term_code"].append(soln.termination_code)
                pygranso_result_summary["iter"].append(soln.iters)
            except Exception as e:
                print_and_log("pygranso failed", log_file, mode="w")
                pygranso_result_summary["data_idx"].append(data_idx)
                pygranso_result_summary["restart_idx"].append(restart_idx)

                pygranso_result_summary["time"].append(-1)
                pygranso_result_summary["mean_error"].append(-1)
                pygranso_result_summary["F"].append(-1)
                pygranso_result_summary["tv"].append(-1)
                pygranso_result_summary["MF"].append(-1)
                pygranso_result_summary["MF_tv"].append(-1)
                pygranso_result_summary["term_code"].append(-1)
                pygranso_result_summary["iter"].append(-1)


    save_dict_to_csv(
            pygranso_result_csv_dir, granso_continue_csv_dir
        )

print("Done")




# n = 10 # V: n*d
# d = 5 # copnst: d*d

# folding_list = ['l2','l1','linf']
# folding_list = ['l2']
# folding_list = ['l2','l1','linf','unfolding']
# folding_list = ['unfolding']

# K = 1000 # K number of starting points. random generated initial guesses
# N = 8 # number of different data matrix (with the same size)
# K = 2
# N = 3


# opt_tol = 1e-16
# maxit = 50000
# maxclocktime = 30
# QPsolver = "gurobi"
# QPsolver = "osqp"
# threshold = 0.99

# square_flag = True
# square_flag = False

# mu0 = 0.1
# device = torch.device('cuda')


###############################################

# [my_path, log_name, date_time, name_str] = utils.get_name(square_flag,folding_list,n,d,K,maxclocktime,N,K)

# if not debug_mode:
#     sys.stdout = open(os.path.join(my_path, log_name), 'w')

###################################################
start_all_seeds = time.time()
# result_dict = utils.result_dict_init(N,folding_list) # initialize result dict
for rng_seed in range(N):
    # [A, U, ana_sol] = utils.data_init(rng_seed, n, d, device)
    folding_idx = 0 # index for plots
    for maxfolding in folding_list:
        print('\n\n\n'+ maxfolding + '  start!')
        folding_idx+=1
        comb_fn = lambda X_struct : utils.user_fn(X_struct,A,d,device,maxfolding,square_flag)
        start_loop = time.time()
        for i in range(K):
            print("the {}th test out of K = {} initial guesses.\n rng seed {} out of N = {}. \n folding type: {} ".format(i+1,K, rng_seed,N,maxfolding))
            try:
                # call pygranso
                start = time.time()
                var_in = {"V": [n,d]}
                opts = utils.opts_init(device,maxit,opt_tol,maxclocktime,QPsolver,mu0,ana_sol,threshold,n,d)
                soln = pygranso(var_spec = var_in,combined_fn = comb_fn,user_opts = opts)
                end = time.time()
                result_dict = utils.store_result(soln,end,start,n,d,i,result_dict,U,rng_seed,maxfolding)
            except Exception as e:
                print('skip pygranso')
        end_loop = time.time()
        print("K Loop Wall Time for {} folding: {}s".format(maxfolding,end_loop - start_loop))
        utils.sort_result(result_dict,rng_seed,maxfolding)
        arr_len = utils.print_result(result_dict,K,rng_seed,maxfolding)
        # plot
        dict_key = str(rng_seed) + maxfolding
        plt.plot(np.arange(arr_len),result_dict[dict_key]['F'],color = colors[folding_idx], marker = markers[folding_idx], linestyle = '-',label=maxfolding)

    plt.plot(np.arange(arr_len),np.array(arr_len*[ana_sol]),color = colors[folding_idx+1], linestyle = '-',label='analytical sol')
    plt.legend()
    plt.title(name_str)
    plt.xlabel('sorted sample index')
    plt.ylabel('obj val')
    

    if not debug_mode:
        [data_name,png_title] = utils.add_path(my_path,rng_seed, date_time, name_str)
        np.save(data_name,result_dict)
        plt.savefig(os.path.join(my_path, png_title))
        plt.clf()
    else:
        plt.show()

end_all_seeds = time.time()
print("N seeds Wall Time: {}s".format(end_all_seeds - start_all_seeds))

if not debug_mode:
    # end writing
    sys.stdout.close()


