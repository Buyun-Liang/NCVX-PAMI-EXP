import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 


def readCSV(csv_file,col_name):

    data = pd.read_csv(csv_file)
    np_data = np.array(data[col_name].tolist())

    return np_data

def get_mean_std(np_data):
    mean = np.mean(np_data)
    std = np.mean(np_data)    
    return [mean,std]



parent_folder = "/home/buyun/Documents/GitHub/NCVX-PAMI-EXP/log_folder/"

file_name = "MSI_Sep22/Trace-Optimization/tr-opt-n10-d5-l2-restart1000-time60"

# summarizing result
csv_file = parent_folder+file_name+"/result_summary.csv"

time = readCSV(csv_file,"time")
[T_mean,T_std] = get_mean_std(time)
total_time = np.sum(time)/3600

data_idx = readCSV(csv_file,"data_idx")
restart_idx = readCSV(csv_file,"restart_idx")

F = readCSV(csv_file,"F")

F_data0 = F[data_idx==0]
analytical_sol = -9.376143


#generate a list of markers and another of colors 
markers = [ "," , "o" , "v" , "^" , "<", ">", "." ]
colors = ['r','g','b','c','m', 'y', 'k']

arr_len = len(F_data0)

plt.plot(np.arange(arr_len),F_data0,color = colors[1], marker = markers[1], linestyle = '-',label="n10d5")

plt.plot(np.arange(arr_len),np.array(arr_len*[analytical_sol]),color = colors[0], linestyle = '-',label='analytical sol')
plt.legend()
plt.title("n10d5")
plt.xlabel('sorted sample index')
plt.ylabel('obj val')
plt.show()

print("Done")
