import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import glob
import h5py
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import time
from datetime import *
from matplotlib.collections import LineCollection
import os, sys


def load_h5_to_dic(fullpath):
    with h5py.File(fullpath, 'r') as file:
        main_keys = list(file["/"].keys())
        data_vector = {}
        if isinstance(file[main_keys[0]], h5py.Dataset):
            #datasets_keys_list = [main_keys]
            for key in main_keys:
                data_vector[key]=file[key][()]
            return data_vector, main_keys
        else:
            datasets_keys_list = {}
            for j, key in enumerate(main_keys):
                datasets_keys = list(file[key].keys())
                datasets_keys_list[key]=list(file[key].keys())
                data_vector[key]={}
                for d_key in datasets_keys:
                    data_vector[key][d_key]=file[key][d_key][()]
            return data_vector, datasets_keys_list 
        
        
def complex_ramsey_fit(t,f,T,phi,A,B):
        Z=A*np.exp(1j*(2*np.pi*f*t+phi))*np.exp(-t/T) + B*(1+1j)
        return np.concatenate([np.real(Z),np.imag(Z)])