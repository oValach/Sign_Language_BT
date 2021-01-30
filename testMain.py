from lib.BP_lib import *
from lib import data_comp
from lib import BP_lib
from timeit import default_timer as timer
import pickle as pk

if __name__ == "__main__":
    DTW = compare_all()

    pk_out = open('DTW.pickle', 'wb')
    pk.dump(DTW, pk_out)
    pk_out.close()