import os
import io
#import ase
import numpy as np
from tabulate import tabulate
#import matplotlib.pyplot as plt


for i in range(28):
    infile  = "final_neb."+str(i)
    outfile = "neb."+str(i)
    with open(infile) as f:
        lines_list = f.readlines()
        lines_list2 = lines_list[15:144]
        my_data = [[float(val) for val in line.split()] for line in lines_list2]
        my_data.sort(key=lambda x: x[0])
        #print(tabulate(my_data))
        #exit()
        f.close()
    #new_data = [[j,j+1] for j in range(len(my_data))]
    new_data = np.array([[my_data[j][2]+my_data[j][5]*11.45087,my_data[j][3]+my_data[j][6]*11.45087,my_data[j][4]+my_data[j][7]*11.45087] for j in range(len(my_data))])
    with open(outfile,'w') as f2:
       new_data.tofile(f2,sep=" ")        
