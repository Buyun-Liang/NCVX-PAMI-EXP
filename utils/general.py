import json

def load_json(json_path):
    content = json.load(open(json_path, "r"))
    return content

def write_log_txt(file_name, msg, mode="a"):
    """
        Write training msg to file in case of cmd print failure in MSI system.
    """
    with open(file_name, mode) as f:
        f.write(msg)
        f.write("\n")

def print_and_log(msg, log_file_name, mode="a", terminal_print=True):
    """
        Write msg to a text file.
    """
    if type(msg) == str:
        if terminal_print == True:
            print(msg)
        write_log_txt(log_file_name, msg, mode=mode)
    elif type(msg) == list:
        for word in msg:
            print_and_log(word, log_file_name, mode=mode, terminal_print=terminal_print)
    else:
        assert RuntimeError("msg input only supports string / List input.")