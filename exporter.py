import numpy as np
import pickle as pk
import os
from lib import bvh2glo_simple
from lib import SL_dict
import matplotlib.pyplot as plt
from lib import data_comp
import operator
import math

BVH_file = 'C:/Users/User/Work/BP/Projekt/data_bvh/16_05_20_a_R.bvh'
dictionary_file = 'C:/Users/User/Work/BP/Projekt/data/ultimate_dictionary2.txt'

joints, trajectory = bvh2glo_simple.calculate(BVH_file)
frames, joint_id, channels = np.shape(trajectory)

dictionary = SL_dict.search_take_file(dictionary_file, BVH_file)

file = open("dict_test2.txt", "a")
file.write(str(dictionary))

pk_out = open('dictionary.pickle', 'wb')
pk.dump(dictionary, pk_out)
pk_out.close()

pk_out = open('trajectory.pickle', 'wb')
pk.dump(trajectory, pk_out)
pk_out.close()
