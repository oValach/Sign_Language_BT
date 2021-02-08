from lib.BP_lib import *
from lib import data_comp
from lib import BP_lib
from timeit import default_timer as timer
import pickle as pk
import numpy as np
import math

if __name__ == "__main__":

    path_bvh = 'Sign_Language_BP/data_bvh'
    path_datatrajectory = 'Sign_Language_BP/data_trajectory'
    path_jointlist = 'Sign_Language_BP/data/joint_list.txt'
    path_converted = 'Sign_Language_BP/data_converted'
    """
    times = timer()
    DTW = compare_all(path_bvh,path_datatrajectory,path_jointlist, path_converted)
    timee = timer()
    print(timee-times)

    pk_out = open('Sign_Language_BP/output_files/DTW.pickle', 'wb')
    pk.dump(DTW, pk_out)
    pk_out.close()

    """
    pkl_in = open("Sign_Language_BP/output_files/DTW.pickle", "rb")
    dtw_out = pk.load(pkl_in)

    pkl_in = open("Sign_Language_BP/output_files/out_words.pickle", "rb")
    out_words = pk.load(pkl_in)

    for key, val in dtw_out[0].items():
        for i in range(len(val)):
            for j in range(len(val[0])):
                if val[i][j] == 0:
                    val[i][j] = math.inf
    
    file = open("Sign_Language_BP/output_files/dtw_sorted.txt", "w")

    for key, val in dtw_out[0].items():
        counter = 0
        val = val.tolist()
        file.write('\n' + str(key) + '\n\n')
        equal = 0
        nonequal = 0
        while(counter <= 100):
            counter += 1
            minValArr = np.amin(val, axis=1)
            minVal = np.amin(minValArr)
            minIdx1 = np.where(minValArr == minVal)[0][0]
            minIdx2 = np.where(val[minIdx1] == minVal)[0][0]
            val[minIdx1][minIdx2] = math.inf
            if out_words[minIdx1] == out_words[minIdx2]:
                equal += 1
            elif out_words[minIdx1] != out_words[minIdx2]:
                nonequal += 1

            try:
                val[minIdx2][minIdx1] = math.inf
            except:
                pass

            file.write(str(out_words[minIdx1]) + ',' + str(out_words[minIdx2]) + ' : ' + str(minVal) + '    '+str(out_words[minIdx1]==out_words[minIdx2]) + '\n')
            if counter == 101:
                file.write(str(equal) + '/' + str(nonequal) + '\n')