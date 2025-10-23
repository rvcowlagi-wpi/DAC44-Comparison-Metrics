import glob
import os
import numpy as np

all_files = glob.glob(os.path.join("Data/lti_1d_trajectories", "*.csv"))

data = []
for file in all_files:
    data.append(np.loadtxt(file))
data1D = np.array(data)

all_files = glob.glob(os.path.join("Data/lti_2d_trajectories", "*.csv"))

data = []
for file in all_files:
    data.append(np.loadtxt(file))
data2D = np.array(data)

np.savetxt('Data/data1D.txt', data1D, delimiter=',')
np.savetxt('Data/data2D.txt', data2D, delimiter=',')