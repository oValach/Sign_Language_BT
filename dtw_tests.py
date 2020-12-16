import os
import math
import operator
import pandas as pd
import numpy as np
import cython
import matplotlib.pyplot as plt
from dtw import dtw
from fastdtw import fastdtw
from datetime import datetime
from collections import OrderedDict
from scipy.spatial.distance import euclidean
from lib import data_comp
from lib import importer

from timeit import timeit

a = [1,1,1]
b = [2,2,2]

a_np = np.array(a)
b_np = np.array(b)

"""
start_time = datetime.now()
dtw_test = dtw(a_np, b_np, dist = euclidean)
end_time = datetime.now()
"""

print(timeit(dtw(a_np, b_np, dist = euclidean)))
print(dtw_test[0])

"""
start_time = datetime.now()
fastdtw_test = fastdtw(a_np, b_np, dist=euclidean)
end_time = datetime.now()
"""

print(timeit(fastdtw(a_np, b_np, dist=euclidean)))
print(fastdtw_test[0])