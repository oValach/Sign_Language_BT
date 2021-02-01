import numpy as np
import pickle as pk
import os
from lib import bvh2glo_simple
from lib import SL_dict
import matplotlib.pyplot as plt
from lib import data_comp
import operator
import math

path = 'Sign_Language_BP/data_bvh'
file_list = os.listdir(path)
file_list = [f for f in file_list if ('bvh' in f)]

done_files = 0
for filepath in file_list:  # iterování přes jednotlivé soubory
    BVH_file = 'Sign_Language_BP/data_bvh/' + filepath
    dictionary_file = 'Sign_Language_BP/data/ultimate_dictionary2.txt'

    joints, trajectory = bvh2glo_simple.calculate(BVH_file)

    dictionary = SL_dict.search_take_file(dictionary_file, BVH_file)
    
    pk_out = open('Sign_Language_BP/data_converted/dictionary_'+filepath[0:12]+'.pickle', 'wb')
    pk.dump(dictionary, pk_out)
    pk_out.close()

    pk_out = open('Sign_Language_BP/data_converted/trajectory_'+filepath[0:12]+'.pickle', 'wb')
    pk.dump(trajectory, pk_out)
    pk_out.close()