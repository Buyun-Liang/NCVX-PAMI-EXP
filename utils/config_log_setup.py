import platform, os, json, torch, random
import numpy as np

# ==== functions related to save log ====
def makedir(dir):
    os.makedirs(dir, exist_ok=True)

def create_log_info(dir, name="experiment_log.txt"):
    log_file = os.path.join(
        dir,
        name
    )
    return log_file

def save_exp_info(save_dir, config):
    """
        Create new log folder / Reuse the checkpoint folder for experiment.
    """
    # Save Exp Settings as Json File
    exp_config_file = os.path.join(save_dir, "Exp_Config.json")
    save_dict_to_json(config, exp_config_file)

def save_dict_to_json(dict, save_dir):
    with open(save_dir, "w") as outfile:
        json.dump(dict, outfile, indent=4)

# ==== functions related to config setup ====
def clear_terminal_output():
    system_os = platform.system()
    if "Windows" in system_os:
        cmd = "cls"
    elif "Linux" in system_os:
        cmd = "clear"
    os.system(cmd)

def set_random_seeds(config):
    """
        This function sets all random seed used in this experiment.
        For reproduce purpose.
    """
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_default_device(config):
    """
        This function set the default device to move torch tensors.

        If GPU is availble, default_device = torch.device("cuda:0").
        If GPU is not available, default_device = torch.device("cpu").
    """
    gpu_list = set_devices(config)
    if gpu_list is None:
        return torch.device("cpu"), gpu_list
    else:
        return torch.device(gpu_list[0]), gpu_list

def set_devices(config):
    """
        This function returns a list of available gpu devices.

        If GPUs do noe exist, return: None
    """
    n_gpu = torch.cuda.device_count()
    print("Total GPUs availbale: [%d]" % n_gpu)
    if n_gpu > 0:
        gpu_list = ["cuda:{}".format(i) for i in range(n_gpu)]
    else:
        gpu_list = None
    set_random_seeds(config)
    return gpu_list