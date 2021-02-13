import os
import math
import operator
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from dtaidistance import dtw
from dtaidistance import dtw_ndim
from dtw import dtw as dtw_slower
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from datetime import datetime
from collections import OrderedDict
from scipy.spatial.distance import euclidean
from lib import data_comp
from lib import BP_lib
from timeit import default_timer as timer

if __name__ == "__main__":

    path_bvh = 'Sign_Language_BP/data_bvh'
    path_datatrajectory = 'Sign_Language_BP/data_trajectory'
    path_jointlist = 'Sign_Language_BP/data/joint_list.txt'
    path_converted = 'Sign_Language_BP/data_converted'
    glo_dir = 'Sign_Language_BP/source_data/'

    testnaslovech = False
    if testnaslovech:
        word1 = 'bude'
        word2 = 'prset-neprset-prset'
        [word1_traj, _, _, _] = BP_lib.find_word(word1, 1,path_bvh,path_converted)
        [word2_traj, _, _, _] = BP_lib.find_word(word2, 1,path_bvh,path_converted)

        #word = 'bude'
        #words_found = BP_lib.find_word(word, 2)
        #[word1_traj,_,_,_] = words_found[0]
        #[word2_traj,_,_,_] = words_found[1]
        start_time = datetime.now()
        dtw_result = BP_lib.dtw_dist(word1_traj, word2_traj,path_jointlist)
        end_time = datetime.now()

        print(str(alg_type) + ': ' + str(word1) + ' vs ' + str(word1))
        [print(str(k)+': '+str(v)) for k, v in dtw_result.items()]
        print('Duration: {}'.format(end_time - start_time))

    testnaumelychpolich_1D = False
    if testnaumelychpolich_1D:
        s1 = np.array(np.random.rand(3000), dtype=np.double)
        s2 = np.array(np.random.rand(3000), dtype=np.double)

        start = timer()
        dtw_test2 = dtw.distance_fast(s1, s2, use_pruning=True)
        #dtw_test2 = dtw.distance_fast(s1, s2, use_pruning=False)
        end = timer()

        print(dtw_test2)
        print(end-start)

    testnaumelychpolich_2D = True
    if testnaumelychpolich_2D:
        np.random.seed(10)
        delka = np.random.randint(15, 40)
        delka2 = np.random.randint(15, 40)

        a = np.zeros((3, delka),dtype=np.double)
        b = np.ones((3, delka),dtype=np.double)

        start = timer()
        dist = dtw_ndim.distance_fast(a,b) #python dtaidistance
        #dist = dtw_slower(a,b, dist=euclidean) #python dtw
        #dist  = fastdtw(a,b,dist=euclidean) #python fastdtw
        end = timer()

        print('{}, čas: {}'.format(dist, end-start))