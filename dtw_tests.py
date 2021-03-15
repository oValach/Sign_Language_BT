import os
import math
import operator
import pandas as pd
import numpy as np
import random
import pickle as pkl
import similaritymeasures
import matplotlib.pyplot as plt
from dtw import dtw as dtw_slower
from dtaidistance import dtw
from dtaidistance import dtw_ndim
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
        meta=traj=[] #pouze pro spravny pruchod find_word
        word1 = 'bude'
        word2 = 'prset-neprset-prset'
        [word1_traj, _, _, _] = BP_lib.find_word(word1, 1,path_bvh,path_converted,meta,traj)
        [word2_traj, _, _, _] = BP_lib.find_word(word2, 1,path_bvh,path_converted,meta,traj)

        start_time = datetime.now()
        dtw_result = BP_lib.dtw_dist(word1_traj, word2_traj,path_jointlist)
        end_time = datetime.now()

        print(str(str(word1) + ' vs ' + str(word1)))
        [print(str(k)+': '+str(v)) for k, v in dtw_result.items()]
        print('Duration: {}'.format(end_time - start_time))

    testnaumelychpolich_1D = False
    if testnaumelychpolich_1D:
        s1 = np.array(np.random.rand(3000), dtype=np.double)
        s2 = np.array(np.random.rand(3000), dtype=np.double)

        start = timer()
        #dist = fastdtw(s1, s2, dist=euclidean)[0]
        #dist = dtw.distance_fast(s1, s2, use_pruning=False,only_ub=True)
        dist = dtw_slower(s1,s2)
        end = timer()

        print(dist)
        print(end-start)

    testnaumelychpolich_2D = True
    if testnaumelychpolich_2D:
        #np.random.seed(10)
        delka = np.random.randint(15, 40)
        delka2 = np.random.randint(15, 30)

        a = np.zeros((3, delka),dtype=np.double)
        b = np.ones((3, delka2),dtype=np.double)

        n1 = np.random.randint(-3,10, size=(3, delka))
        n2 = np.random.randint(-1,12, size=(3, delka2))

        start = timer()
        #dist0 = dtw_ndim.distance_fast(np.transpose(a),np.transpose(b),only_ub=True)  #python dtaidistance
        #dist1 = dtw_slower(np.transpose(a),np.transpose(b)).distance    #python dtw
        #dist2  = fastdtw(np.transpose(a),np.transpose(b),dist=euclidean)[0]    #python fastdtw
        dist3 = similaritymeasures.area_between_two_curves(n1, n2) #computing area between 2 trajetories - frechet computes only max length of the connection
        end = timer()

        print('{}, čas: {}'.format(dist3, end-start))