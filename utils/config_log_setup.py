import platform
import os

# ==== functions related to config setup ====
def clear_terminal_output():
    system_os = platform.system()
    if "Windows" in system_os:
        cmd = "cls"
    elif "Linux" in system_os:
        cmd = "clear"
    os.system(cmd)