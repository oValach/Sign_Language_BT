from data_mining import load, compute

paths = 'data/paths_.txt'

bvh_dir,bvh_dict,source_dir,path_jointlist,path_chosen_joints,path_dictionary,path_metadata,path_trajectory,path_output,joint_list,meta,traj = load(paths)

alg_type = 'softdtw'

distance_matrix = compute(path_output, path_trajectory, path_chosen_joints, alg_type=alg_type, graph = 1, word_amount=-1)
