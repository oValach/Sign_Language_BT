import os
import math
import operator
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from dtaidistance import dtw as dtw
from fastdtw import fastdtw
from datetime import datetime
from collections import OrderedDict
from scipy.spatial.distance import euclidean
from lib import data_comp
from lib import BP_lib
from timeit import timeit

if __name__ == "__main__":
    """
    #TEST NA SLOVECH
    word1 = 'bude'
    word2 = 'prset-neprset-prset'
    [word1_traj,_,_,_] = BP_lib.find_word(word1, 1)
    [word2_traj,_,_,_] = BP_lib.find_word(word2, 1)
 
    #word = 'bude'
    #words_found = BP_lib.find_word(word, 2)
    #[word1_traj,_,_,_] = words_found[0]
    #[word2_traj,_,_,_] = words_found[1]
    start_time = datetime.now()
    dtw_result = BP_lib.dtw_dist(word1_traj, word2_traj)
    end_time = datetime.now()

    print(str(alg_type) + ': ' + str(word1) + ' vs ' + str(word1))
    [print(str(k)+': '+str(v)) for k, v in dtw_result.items()]
    print('Duration: {}'.format(end_time - start_time))
    """

    #TEST NA UMELYCH POLICH
    s1 = np.array(np.random.rand(3000), dtype=np.double)
    s2 = np.array(np.random.rand(3000), dtype=np.double)

    start_time = datetime.now()
    dtw_test2 = dtw.distance_fast(s1,s2,use_pruning=True)
    #dtw_test2 = dtw.distance_fast(s1, s2, use_pruning=False)
    end_time = datetime.now()

    print(dtw_test2)
    print(end_time-start_time)