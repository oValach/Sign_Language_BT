from data_mining import *

#paths = 'data/paths.txt' # for metacentrum calculations
paths = 'Sign_Language_BP/data/paths.txt'

with open(paths, 'r') as pth:
    paths_list = pth.readlines()
    
bvh_dir = paths_list[0].rstrip("\n") # all bvh files takes and dictionaries
bvh_dict = paths_list[1].rstrip("\n") # bvh files with separate words signed
source_dir = paths_list[2].rstrip("\n") # data converted from angular to global positions
path_jointlist = paths_list[3].rstrip("\n") # path to the joint_list.txt file 
path_chosen_joints = paths_list[4].rstrip("\n") # path to the chosen joints indexes from joint_list.txt file
path_dictionary = paths_list[5].rstrip("\n") # path to the ultimate_dictionary2.txt file
path_metadata = paths_list[6].rstrip("\n") # path to the meta.pkl file
path_trajectory = paths_list[7] # path to the traj.pkl file

with open(path_jointlist, 'r') as f:
    joint_list = f.readlines()      # the list of markers (tracked body parts)
with open(path_metadata, 'rb') as pf:
    meta = pk.load(pf)              # metadata: meaning, the initial file, anotation
with open(path_trajectory, 'rb') as pf:
    traj = pk.load(pf)              # trajektory [item, frame, joint, channel]

alg_type = 'method_combination'
resample_method = 'interpolation'
int_method = 'linear'
distance_method = 'euclidean'

distance_matrix = compute(path_jointlist, path_trajectory, path_chosen_joints, alg_type=alg_type, order='toLonger', resample_method=resample_method, int_method=int_method, distance_method=distance_method, graph = 1, word_amount=1)
