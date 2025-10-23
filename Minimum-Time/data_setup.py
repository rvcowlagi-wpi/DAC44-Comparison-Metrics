import glob
import os
import numpy as np

os.remove("Data/data_mintime.txt")
all_files = glob.glob(os.path.join("Data/Variable-Wind-Constant-Endpoints/F175", "*.csv"))

data = []
for file in all_files:
    data.append(np.loadtxt(file))
data_mintime = np.array(data)


np.savetxt('Data/data_mintime.txt', data_mintime, delimiter=',')
