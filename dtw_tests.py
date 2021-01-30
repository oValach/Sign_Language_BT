import os
import math
import operator
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from dtw import dtw
from fastdtw import fastdtw
from datetime import datetime
from collections import OrderedDict
from scipy.spatial.distance import euclidean
from lib import data_comp
from lib import BP_lib
from timeit import timeit

if __name__ == "__main__":
    
    #TEST NA SLOVECH
    word1 = 'bude'
    word2 = 'prset-neprset-prset'
    [word1_traj,_,_,_] = BP_lib.find_word(word1, 1)
    [word2_traj,_,_,_] = BP_lib.find_word(word2, 1)
 
    #word = 'bude'
    #words_found = BP_lib.find_word(word, 2)
    #[word1_traj,_,_,_] = words_found[0]
    #[word2_traj,_,_,_] = words_found[1]
    alg_type = 'fastdtw'
    start_time = datetime.now()
    dtw_result = BP_lib.dtws(alg_type, word1_traj, word2_traj)
    end_time = datetime.now()

    print(str(alg_type) + ': ' + str(word1) + ' vs ' + str(word1))
    [print(str(k)+': '+str(v)) for k, v in dtw_result.items()]
    print('Duration: {}'.format(end_time - start_time))

    alg_type = 'dtw'
    start_time = datetime.now()
    dtw_result = BP_lib.dtws(alg_type, word1_traj, word2_traj)
    end_time = datetime.now()

    print(str(alg_type) + ': ' + str(word1) + ' vs ' + str(word2))
    [print(str(k)+': '+str(v)) for k, v in dtw_result.items()]
    print('Duration: {}'.format(end_time - start_time))

    #TEST NA UMELYCH POLICH
    a = [1,1,1]
    b = [2,2,2]

    start_time = datetime.now()
    dtw_test = dtw(a, b, dist=euclidean)
    end_time = datetime.now()
    print(dtw_test[0])

    start_time = datetime.now()
    fastdtw_test = fastdtw(a, b, dist=euclidean)
    end_time = datetime.now()
    print(fastdtw_test[0])