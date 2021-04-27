from data_mining import *

#paths = 'data/paths.txt' # for metacentrum calculations
paths = 'Sign_Language_BP/data/paths.txt'

bvh_dir,bvh_dict,source_dir,path_jointlist,path_chosen_joints,path_dictionary,path_metadata,path_trajectory,joint_list,meta,traj = load(paths)

alg_type = 'method_combination'
resample_method = 'interpolation'
int_method = 'linear'
distance_method = 'euclidean'
order = 'toShorter'

distance_matrix = compute(path_trajectory, path_chosen_joints, alg_type=alg_type, order=order, resample_method=resample_method, int_method=int_method, distance_method=distance_method, graph = 1, word_amount=-1)
